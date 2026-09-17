"""Standalone checks for YAY-Robot foresight correction (no VLM, no CARLA, no JAX).

    JAX_PLATFORMS=cpu WANDB_MODE=disabled PYTHONPATH=impls uv run python \\
        impls/coaches/test_yay_robot.py

Covers what separates this path from cast_relabel: the correction is decided at CoT-query time and
is stored together with the ``pre_intervention_sec`` of frames leading up to it, labelled with the
vocabulary ``SteerVLAActor.update_hl`` buckets on. Also covers the episode-level strategy bank that
replaced the old per-correction CorrectionMemory.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coaches import yay_robot as yr

FAILURES: list[str] = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  -- ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


CORRECT = json.dumps(
    {
        "verdict": "CORRECT",
        "rationale": "Light is green and the way is clear.",
        "corrected_subtask": "The vehicle accelerates normally through the green traffic light.",
        "corrected_reasoning": "Follow the route. Accelerate because the traffic light ahead is green.",
    }
)
KEEP = json.dumps(
    {"verdict": "KEEP", "rationale": "fine", "corrected_subtask": "", "corrected_reasoning": ""}
)
STRATEGY = "Crept to every junction and scored 27; take plainly available gaps sooner."


class StubCoach:
    """Answers CORRECT on nominated calls, KEEP otherwise; summarises without a video."""

    def __init__(self):
        self.calls = 0
        self.correct_next = False
        self.last_prompt = ""
        self.strategy_prompt = ""

    def complete_image_text(self, image, prompt):
        self.calls += 1
        self.last_prompt = prompt
        return CORRECT if self.correct_next else KEEP

    def complete_text(self, prompt):
        self.strategy_prompt = prompt
        return STRATEGY

    def analyze_video_text(self, video_path, prompt):
        self.strategy_prompt = prompt
        return STRATEGY


def make_session(tmp, **over):
    cfg = {
        "enabled": True,
        "query_every_n_cot_queries": 1,
        "pre_intervention_sec": 2.0,
        "env_step_dt": 0.05,
        "pre_intervention_stride_steps": 10,
        "max_pre_intervention_samples": 8,
        "store_kept_samples": True,
        "kept_sample_stride": 4,
        "action_chunk_steps": 10,
        "hl_action_dim": 4,
        "strategy_memory_entries": 8,
        "save_artifacts": True,
        "debug": False,
    }
    cfg.update(over)
    s = yr.OnlineYayRobotSession(cfg, save_dir=tmp, action_chunk_steps=10, run_tag="test")
    s._coach = StubCoach()
    return s


def query(step):
    return {
        "subtask": "The vehicle remains stopped normally due to a red traffic light.",
        "reasoning": "Follow the route. Remain stopped because of the red traffic light.",
        "image": np.full((8, 16, 3), step % 255, dtype=np.uint8),
        "state": np.arange(20, dtype=np.float32),
        "current_speed": 0.0,
        "prompt": "The current speed is 0.0 m/s. Follow the route.",
        "routing_command": "Follow the route.",
    }


def drive(sess, steps, correct_on_steps, query_every=5):
    """One env step per iteration: set_step -> (maybe) CoT query -> record_model_input."""
    for step in range(1, steps + 1):
        sess.set_step(episode_step=step, global_step=step)
        q = query(step)
        out = None
        if step % query_every == 0:
            sess._coach.correct_next = step in correct_on_steps
            out = sess.cot_intervention(q)
        sub = out[0] if out else q["subtask"]
        rea = out[1] if out else q["reasoning"]
        sess.record_model_input(
            episode_step=step,
            image=q["image"],
            state=q["state"],
            current_speed=0.0,
            prompt=q["prompt"],
            subtask=sub,
            reasoning=rea,
            action_chunk=np.zeros((10, 4), dtype=np.float32),
            routing_command=q["routing_command"],
            global_step=step,
        )


def test_intervention_and_leadup(tmp):
    print("\nintervention + 2 s lead-up")
    s = make_session(tmp / "a")
    s.begin_episode(episode_count=1, route_id="test-route-001")
    drive(s, 60, correct_on_steps={50})

    dirs = sorted(p.name for p in s.hl_dataset_dir.iterdir())
    interv = [d for d in dirs if d.endswith("_intervention")]
    check("one intervention dir written", len(interv) == 1, str(dirs))
    if not interv:
        return s
    man = json.loads((s.hl_dataset_dir / interv[0] / "hl_samples.json").read_text())
    steps = sorted(x["episode_step"] for x in man["samples"])
    lead = [x for x in steps if x != 50]
    check("intervention frame stored", 50 in steps, str(steps))
    check("lead-up inside 2 s window", bool(lead) and all(10 <= x < 50 for x in lead), str(lead))
    check(
        "lead-up labelled BAD/precursor",
        all(
            x["credit_source"] == yr.CREDIT_PRECURSOR
            for x in man["samples"]
            if x["episode_step"] != 50
        ),
    )
    direct = [x for x in man["samples"] if x["episode_step"] == 50]
    check(
        "intervention frame is BAD/direct and action-matched",
        bool(direct)
        and direct[0]["credit_source"] == yr.CREDIT_DIRECT
        and direct[0]["action_matches_subtask"] is True,
    )
    check(
        "every sample carries the corrected subtask",
        all(x["subtask"].startswith("The vehicle accelerates") for x in man["samples"]),
    )
    check(
        "HL samples never supervise the action expert",
        all(x["action_supervision"] is False for x in man["samples"]),
    )
    npz = np.load(s.hl_dataset_dir / interv[0] / man["samples"][0]["sample_file"])
    check("action_loss_mask all False", not npz["action_loss_mask"].any())
    check("kept (reinforce) samples written", any(d.endswith("_kept") for d in dirs))
    return s


def test_outcome_backfill(s):
    print("\noutcome backfill")
    interv = [p for p in s.hl_dataset_dir.iterdir() if p.name.endswith("_intervention")]
    outcome = s.finalize_episode(metadata={"termination_reason": "crash_stuck"})
    check("outcome resolved", outcome == "crash_stuck", outcome)
    if interv:
        man = json.loads((interv[0] / "hl_samples.json").read_text())
        check(
            "outcome stamped on written manifests",
            {x["outcome"] for x in man["samples"]} == {"crash_stuck"},
        )


def test_strategy_memory(tmp):
    print("\nepisode strategy memory")
    s = make_session(tmp / "b")
    s.begin_episode(
        episode_count=1,
        route_id="test-route-001",
        route_command_plan=[{"command": "Turn right in 20 meters.", "start_distance_m": 20}],
    )
    drive(s, 40, correct_on_steps={20})
    check("interventions collected for the summary", len(s._episode_corrections) == 1)
    check("executed spans collected", len(s._episode_chunks) >= 2)

    sentence = s.end_episode(driving_score=27.0, route_goal="turn right at the junction")
    check("summary sentence returned", sentence == STRATEGY, sentence)
    check("bank holds the episode", len(s._memory.strategies) == 1)
    check("bank renders a block", "Strategy memory" in s._memory.render())
    check("episode state cleared", not s._episode_corrections and not s._episode_chunks)
    check(
        "strategy prompt saw the intervention",
        "CORRECTION" in s._coach.strategy_prompt or "accelerates" in s._coach.strategy_prompt,
    )
    check(
        "bank persisted to disk",
        (s.artifact_dir / "strategy_memory.json").is_file(),
    )

    # The bank must reach the NEXT episode's foresight prompt -- that is the whole point of it.
    s.begin_episode(episode_count=2, route_id="test-route-001")
    s.set_step(episode_step=1, global_step=100)
    s._coach.correct_next = False
    s.cot_intervention(query(1))
    check("next episode's prompt carries the bank", STRATEGY in s._coach.last_prompt)


def test_parser_and_failure(tmp):
    print("\nparser + failure handling")
    check("KEEP is not an intervention", yr.parse_foresight_correction(KEEP).is_intervention is False)
    check("garbage degrades to KEEP", yr.parse_foresight_correction("not json").is_intervention is False)
    check("CORRECT parses", yr.parse_foresight_correction(CORRECT).is_intervention is True)
    off = json.dumps(
        {"verdict": "CORRECT", "corrected_subtask": "x", "corrected_reasoning": "I must stop now."}
    )
    check(
        "off-template reasoning normalised",
        yr.parse_foresight_correction(off).reasoning.startswith("Follow the route."),
        yr.parse_foresight_correction(off).reasoning,
    )

    s = make_session(tmp / "c", max_consecutive_failures=2)

    class Boom:
        def complete_image_text(self, *a):
            raise RuntimeError("api down")

    s._coach = Boom()
    s.begin_episode(episode_count=1, route_id="r")
    for step in (1, 2):
        s.set_step(episode_step=step, global_step=step)
        check(f"failure {step} degrades to KEEP", s.cot_intervention(query(step)) is None)
    check("session disables itself after repeated failures", s._disabled)
    s.close()


def main():
    tmp = Path(tempfile.mkdtemp())
    s = test_intervention_and_leadup(tmp)
    test_outcome_backfill(s)
    s.close()
    test_strategy_memory(tmp)
    test_parser_and_failure(tmp)
    print(f"\n{'FAILED: ' + ', '.join(FAILURES) if FAILURES else 'ALL CHECKS PASSED'}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
