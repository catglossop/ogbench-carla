"""YAY-Robot: foresight language correction at CoT-query time (no hindsight relabeling).

This is the *online, forward-looking* sibling of :mod:`coaches.cast_relabel`. Both end up
writing SteerVLA high-level (``steervla_hl_dataset_format``) training samples, and both use a
VLM to decide what the policy's subtask/reasoning *should* be -- but they sit on opposite sides
of the action:

``cast_relabel`` (hindsight)
    Drive a window -> review the window video -> assign GOOD/BAD credit to chunks that are
    already history -> relabel them. The corrected language never conditions the policy; it only
    becomes a training target for a later gradient step.

``yay_robot`` (foresight, this module)
    **Every time the model is queried for a CoT**, the freshly sampled subtask/reasoning is shown
    to the VLM together with the current camera frame, *before* the action expert has run. The VLM
    either KEEPs the language or issues a correction. A correction is an **intervention**: the
    corrected subtask/reasoning is re-tokenized and handed to the flow/action expert as its
    conditioning for this query (see ``SteerVLAActor.cot_intervention_fn`` and
    ``SteerVLAActor._build_text_cot_out``), so the vehicle actually drives on the corrected
    language. The same correction is then stored as HL supervision.

Which is to say: CAST asks "what should it have said?", YAY asks "what should it say *now*?" --
and then makes it say that. There is no video, no credit assignment and no window: the unit of
work is one CoT query, one image, one blocking VLM call.

What gets stored (see :meth:`OnlineYayRobotSession.record_model_input`):

* **The intervention frame** -- the step the correction fired on, targeted with the corrected
  subtask + reasoning. Labeled ``BAD`` / ``credit_source="direct"``.
* **The lead-up** -- the frames covering the ``pre_intervention_sec`` (2 s by default) *before*
  the intervention, also targeted with the corrected language. Labeled ``BAD`` /
  ``credit_source="precursor"``. This is the point of doing it in foresight: if the VLM had to
  correct the language at step S, the policy was already drifting toward the wrong subtask for
  some time before S, and those earlier frames are where the correction is actually actionable.
* **Kept frames** (``store_kept_samples``) -- when the VLM leaves the language alone, the frame is
  stored with the model's *own* subtask/reasoning, labeled ``GOOD``. This is the reinforce path,
  exactly like ``cast_relabel.store_good_chunks``.

The ``BAD``/``GOOD`` and ``direct``/``precursor`` vocabulary is deliberately borrowed verbatim
from ``cast_relabel`` rather than invented fresh, because ``SteerVLAActor.update_hl`` buckets the
online pool on exactly those two strings (``hl_online_bad_fraction`` /
``hl_online_precursor_fraction`` / ``use_adaptive_sampling``). Reusing them means a YAY pool drops
into the existing sampler unchanged; the provenance specific to this method lives in the
per-intervention ``yay_robot.json`` artifacts.

Samples are written the moment they are produced -- the run consumes them online, and
``steervla.hl_checkpoint_every_steps=2000`` redeploys the fine-tuned backbone every 2k env steps --
so the episode ``outcome`` tag is not known yet when a manifest is written.
:meth:`finalize_episode` patches it into the already-written manifests at episode end; that is a
small JSON rewrite, no ``.npz`` is touched.
"""

from __future__ import annotations

import json
import os
import textwrap
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from coaches.action_chunk_feedback import DEFAULT_ACTION_CHUNK_STEPS
from coaches.cast_relabel import (
    DEFAULT_EGO_HISTORY_LEN,
    DEFAULT_HL_ACTION_DIM,
    EGO_STATE_IDX_SPEED,
    EGO_STATE_IDX_YAW_RATE,
    SEED_REASONING,
    SEED_SUBTASKS,
    SIMLINGO_FRAME_DT,
    HLSample,
    _ego_hist_from_state,
    _extract_json_payload,
    _shape_hl_action_chunk,
    normalize_reasoning,
    resolve_window_outcome,
    strip_cot_sentinels,
    write_hl_samples,
)
from coaches.correction_memory import DEFAULT_MAX_WORDS as DEFAULT_MEMORY_WORDS
from coaches.correction_memory import CorrectionMemory
from coaches.vlm_feedback import create_coach

# One CARLA env step at the 20 Hz the leaderboard wrapper ticks. Only used to convert
# ``pre_intervention_sec`` into a number of env steps; override with ``env_step_dt``.
DEFAULT_ENV_STEP_DT = 0.05

# How far back the corrected language is propagated, in seconds. The intervention itself is one
# frame; this is the lead-up that gets the same target, so the policy learns to reach the
# corrected subtask *before* the VLM has to ask for it.
DEFAULT_PRE_INTERVENTION_SEC = 2.0

# Labels borrowed from cast_relabel so ``SteerVLAActor.update_hl`` buckets these samples with its
# existing corrective/reinforce split. See the module docstring.
LABEL_INTERVENTION = "BAD"
LABEL_KEPT = "GOOD"
CREDIT_DIRECT = "direct"
CREDIT_PRECURSOR = "precursor"


@dataclass(frozen=True)
class CotCorrection:
    """One VLM verdict on a freshly sampled CoT."""

    changed: bool
    subtask: str
    reasoning: str
    rationale: str = ""
    # Raw model text, kept for the artifact so a bad parse stays debuggable.
    raw_response: str = ""

    @property
    def is_intervention(self) -> bool:
        return bool(self.changed and self.subtask and self.reasoning)


def build_foresight_correction_prompt(
    *,
    subtask: str,
    reasoning: str,
    routing_command: str,
    telemetry: dict[str, Any],
    subtask_history: list[str] | None = None,
    memory_block: str = "",
    seed_subtasks: tuple[str, ...] = SEED_SUBTASKS,
    seed_reasonings: tuple[str, ...] = SEED_REASONING,
    max_seed_examples: int = 48,
) -> str:
    """The foresight analogue of :func:`cast_relabel.build_credit_relabel_prompt`.

    Same driving priorities, same subtask/reasoning vocabulary and the same hard reasoning
    template -- but the question is about the *next* couple of seconds rather than a window
    already driven, so there are no events, no chunks and no credit assignment. The VLM sees one
    frame, the ego telemetry, and the CoT the policy just produced for it.
    """
    seed_subtask_block = "\n".join(f"- {s}" for s in list(seed_subtasks)[:max_seed_examples])
    seed_reasoning_block = "\n".join(f"- {s}" for s in list(seed_reasonings)[:max_seed_examples])
    history_block = ""
    if subtask_history:
        recent = "\n".join(f"- {s}" for s in subtask_history)
        history_block = (
            "\nThe subtasks the policy produced over the preceding few seconds, oldest first. Use\n"
            "them to judge whether the current one is a sensible continuation or an abrupt,\n"
            "unjustified switch:\n" + recent + "\n"
        )

    return textwrap.dedent(
        f"""
        You are supervising a driving policy in real time. The attached image is the vehicle's
        current forward camera view. The policy has just produced the chain-of-thought below for
        THIS frame, and is about to hand it to a low-level controller that will drive the next
        couple of seconds on it. Your job is to decide, BEFORE it drives, whether that language
        describes the right thing to do next -- and if it does not, to replace it.

        This is a FORESIGHT judgement, not a review. Nothing has happened yet. Do not describe or
        grade past behaviour; decide what the vehicle should do from this frame onward.

        PRIORITY. **Completing the route is the primary objective.** Safety and traffic rules are
        constraints on how the vehicle makes progress, not goals in their own right. Language that
        has the vehicle sit still, crawl, or keep waiting when the way ahead is clear is WRONG and
        must be corrected -- over-conservatism is as real a defect as recklessness. Correct toward
        movement (take the gap, complete the turn, resume speed) whenever both a progress-making
        and a further-slowing instruction would be defensible. Only correct toward stopping or
        slowing when you can name the specific hazard or signal VISIBLE IN THIS FRAME that
        requires it.

        The policy's CoT is its own output and is frequently WRONG about the scene -- most
        commonly it claims a red traffic light when the light in the image is green, or names a
        hazard that is not there. The image is what settles the scene; never treat the CoT as
        evidence about it. Where the CoT contradicts the image, that is a correction, and the
        replacement must describe what the image actually shows.

        Routing command (the instruction the vehicle is following): {routing_command or "Follow the route."}

        Current ego telemetry:
        ```json
        {json.dumps(telemetry, indent=2)}
        ```

        The policy's proposed chain-of-thought for this frame:
        ```json
        {json.dumps({"subtask": subtask, "reasoning": reasoning}, indent=2)}
        ```
        {history_block}{memory_block}
        Example subtask phrasings (open vocabulary -- reuse verbatim OR write new phrases in the
        SAME concise style; describe what the vehicle should do, not meta commentary). HOWEVER the
        corrected subtask should NOT reference objects in the scene and should focus on the driving
        behaviour of the vehicle, to prevent out-of-distribution objects from destroying the
        training signal.:
        {seed_subtask_block}

        REQUIRED REASONING FORMAT. ``corrected_reasoning`` MUST follow the template below -- this
        is a hard constraint, not a stylistic preference. Reasonings that deviate are discarded:

          "Follow the route. <ACTION> <because|to> <justification referencing the scene>."

        Rules:
          1. Begin with the literal prefix "Follow the route." -- always, with no preamble.
          2. Follow it with ONE imperative action clause, in the same vocabulary as the examples:
             Accelerate / Maintain the speed / Maintain speed / Decelerate / Remain stopped /
             Slow down / Stop.
          3. Then justify it with "because ..." or "to ...", naming the relevant object in the
             image (colour + type + position, e.g. "the maroon car that is to the front at 8.0
             meters") when one is relevant.
          4. NEVER write in the first person. No "I", "my", "I must", "I should".
          5. NEVER lead with the explanation, and never use contrastive framing ("Even though ...").
          6. One sentence after the prefix.

        Example reasoning phrasings (reuse verbatim where one fits, otherwise write a new phrase
        that obeys the template above):
        {seed_reasoning_block}

        Unlike the subtask, ``corrected_reasoning`` MAY reference objects in the scene.

        DO NOT USE "reverse" or "back up" or "backwards" OR ANYTHING LIKE THIS in the corrected
        subtask. The model cannot execute it and it will destroy the training signal.

        Intervene only when it matters. If the proposed subtask and reasoning would produce
        acceptable driving for the next couple of seconds, return KEEP -- even if you would have
        phrased it differently. Rewording for its own sake teaches the policy nothing and costs a
        real intervention its meaning. Return CORRECT when the language would produce the wrong
        behaviour: it contradicts what the image shows, it stops or slows with no visible cause,
        it keeps waiting when the way is clear, or it fails to anticipate a hazard that is already
        visible in this frame.

        Return ONLY valid JSON (no markdown fences):
        {{
          "verdict": "KEEP" or "CORRECT",
          "rationale": "one short sentence saying why",
          "corrected_subtask": "...",
          "corrected_reasoning": "Follow the route. ..."
        }}

        Rules:
        - ``corrected_subtask`` and ``corrected_reasoning`` must be "" when the verdict is KEEP.
        - Both must be non-empty when the verdict is CORRECT.
        - Keep ``rationale`` under 30 words.
        """
    ).strip()


def parse_foresight_correction(text: str) -> CotCorrection:
    """Parse the verdict JSON. A malformed or unparseable response degrades to KEEP.

    Failing closed matters more here than in ``cast_relabel``: a bad parse there loses a window of
    offline labels, whereas here the returned language is about to *drive the car*. KEEP means the
    policy's own CoT is used, which is the pre-existing behaviour.
    """
    raw = str(text or "")
    try:
        payload = _extract_json_payload(raw)
    except Exception:  # noqa: BLE001 - never let a bad response take down the rollout.
        return CotCorrection(changed=False, subtask="", reasoning="", raw_response=raw)

    verdict = str(payload.get("verdict") or "").strip().upper()
    rationale = str(payload.get("rationale") or "").strip()
    if verdict != "CORRECT":
        return CotCorrection(
            changed=False, subtask="", reasoning="", rationale=rationale, raw_response=raw
        )

    subtask = strip_cot_sentinels(payload.get("corrected_subtask"))
    # The same normalisation the CAST relabel path applies: the template is a hard constraint and
    # compliance is never total. It returns "" when the suggestion cannot be made to fit, which is
    # treated here as "no usable correction" rather than shipping off-template language to the
    # model -- both as conditioning and as a training target.
    reasoning = normalize_reasoning(payload.get("corrected_reasoning"))
    if not subtask or not reasoning:
        return CotCorrection(
            changed=False, subtask="", reasoning="", rationale=rationale, raw_response=raw
        )
    return CotCorrection(
        changed=True, subtask=subtask, reasoning=reasoning, rationale=rationale, raw_response=raw
    )


def _as_config_dict(cfg: Any) -> dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return dict(cfg)
    to_dict = getattr(cfg, "to_dict", None)
    return dict(to_dict()) if callable(to_dict) else dict(cfg)


class OnlineYayRobotSession:
    """Foresight CoT correction + intervention-sample collection for one online run.

    Two call sites, and keeping them straight is the whole contract:

    * :meth:`cot_intervention` runs **inside the actor**, on the CoT-query path, and must return
      quickly and never raise -- its return value becomes the flow expert's conditioning. It is
      wired in as ``SteerVLAActor.cot_intervention_fn``.
    * :meth:`set_step`, :meth:`record_model_input` and :meth:`finalize_episode` run in
      ``main_carla``'s rollout loop and do the bookkeeping: which env step the actor is about to
      act on, the rolling buffer of recent frames, and writing the HL samples out.

    They are split that way because the correction is decided *before* the action exists but the
    sample needs the executed action chunk, which only exists *after* the env step. The correction
    is therefore parked in ``_pending`` by the actor-side call and flushed by the loop-side one.
    """

    def __init__(
        self,
        yay_cfg: Any,
        *,
        save_dir: str | Path,
        action_chunk_steps: int = DEFAULT_ACTION_CHUNK_STEPS,
        run_tag: str = "",
    ) -> None:
        self.cfg = _as_config_dict(yay_cfg)
        self.save_dir = Path(save_dir)
        self.artifact_dir = self.save_dir / "yay_robot"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        self.provider = str(self.cfg.get("provider", "gemini"))
        self.gemini_model = str(self.cfg.get("gemini_model", "gemini-3.5-flash"))
        self.action_chunk_steps = max(1, int(self.cfg.get("action_chunk_steps", action_chunk_steps)))
        self.hl_action_dim = int(self.cfg.get("hl_action_dim", DEFAULT_HL_ACTION_DIM))
        self.ego_history_len = max(1, int(self.cfg.get("ego_history_len", DEFAULT_EGO_HISTORY_LEN)))
        self.save_artifacts = bool(self.cfg.get("save_artifacts", True))
        self.debug = bool(self.cfg.get("debug", False))

        # ── what to store ───────────────────────────────────────────────────────────────
        self.store_hl_dataset = bool(self.cfg.get("store_hl_dataset", True))
        # Reinforce path: store KEPT frames with the model's own (uncorrected) CoT, mirroring
        # cast_relabel.store_good_chunks. False -> the pool is corrective (interventions) only.
        self.store_kept_samples = bool(self.cfg.get("store_kept_samples", True))
        # KEPT frames vastly outnumber interventions (most CoTs are fine), so store only every
        # Nth of them or the reinforce bucket swamps the corrective one on disk. The batch-level
        # ratio is still governed by ``steervla.hl_online_bad_fraction``; this only bounds how
        # much gets written.
        self.kept_sample_stride = max(1, int(self.cfg.get("kept_sample_stride", 4)))

        # ── the 2 s lead-up ─────────────────────────────────────────────────────────────
        self.env_step_dt = float(self.cfg.get("env_step_dt", DEFAULT_ENV_STEP_DT))
        self.pre_intervention_sec = float(
            self.cfg.get("pre_intervention_sec", DEFAULT_PRE_INTERVENTION_SEC)
        )
        self.pre_intervention_steps = max(
            0, round(self.pre_intervention_sec / max(1e-6, self.env_step_dt))
        )
        # Only every Nth env step is buffered as a candidate lead-up frame. Defaults to one action
        # chunk, which is the same frame spacing cast_relabel stores (chunk starts only) and keeps
        # the ring buffer to a handful of full-resolution images rather than 40 of them.
        self.pre_intervention_stride_steps = max(
            1, int(self.cfg.get("pre_intervention_stride_steps", self.action_chunk_steps))
        )
        # Hard cap on lead-up samples emitted per intervention, in case the stride is set small.
        self.max_pre_intervention_samples = max(
            0, int(self.cfg.get("max_pre_intervention_samples", 8))
        )

        # ── when to query ───────────────────────────────────────────────────────────────
        # The correction is a blocking VLM call on the action path. ``query_every_n_cot_queries``
        # thins it (1 = every CoT query, the true YAY-Robot regime); ``min_seconds_between_queries``
        # is a wall-clock floor so a fast actor cannot outrun the API's rate limit.
        self.query_every_n_cot_queries = max(1, int(self.cfg.get("query_every_n_cot_queries", 1)))
        self.min_seconds_between_queries = float(self.cfg.get("min_seconds_between_queries", 0.0))
        # Give up on a hung API call rather than stalling the rollout indefinitely. The CARLA
        # watchdog is already paused across the actor forward (main_carla pauses the sim around
        # ``_sample_agent_action``), so this bounds sim-time stall, not just wall clock.
        self.request_timeout_sec = float(self.cfg.get("request_timeout_sec", 30.0))
        # Consecutive VLM failures after which the session stops querying for the rest of the run.
        # Without it a dead API key turns every single CoT query into a timeout.
        self.max_consecutive_failures = int(self.cfg.get("max_consecutive_failures", 5))

        # How many recent subtasks to show the VLM as continuity context.
        self.subtask_history_len = max(0, int(self.cfg.get("subtask_history_len", 4)))

        # ── where the samples land (same convention as cast_relabel) ────────────────────
        self.run_tag = str(run_tag or Path(self.save_dir).name or "run")
        hl_root = str(self.cfg.get("hl_dataset_root", "") or "").strip()
        if hl_root:
            self.hl_dataset_dir = Path(hl_root).expanduser() / self.run_tag
        else:
            self.hl_dataset_dir = self.save_dir / str(
                self.cfg.get("hl_dataset_subdir", "yay_robot_hl_dataset")
            )
        if self.store_hl_dataset:
            self.hl_dataset_dir.mkdir(parents=True, exist_ok=True)

        raw_seeds = self.cfg.get("seed_subtasks")
        self.seed_subtasks: tuple[str, ...] = (
            tuple(str(s) for s in raw_seeds) if raw_seeds else SEED_SUBTASKS
        )

        self._coach = create_coach(self.provider, model=self.gemini_model)
        memory_words = int(self.cfg.get("correction_memory_words", DEFAULT_MEMORY_WORDS))
        # Bounded cross-query memory of corrections already made. Even more necessary here than in
        # cast_relabel: a per-frame stateless judge will otherwise flip the same decision back and
        # forth every couple of seconds and teach the backbone both directions of it.
        self._memory: CorrectionMemory | None = (
            CorrectionMemory(
                self.artifact_dir / "correction_memory.json",
                max_words=memory_words,
                coach=self._coach,
            )
            if memory_words > 0
            else None
        )
        if self.provider == "gemini" and not os.environ.get("GEMINI_API_KEY", ""):
            print(
                "[yay_robot] GEMINI_API_KEY is not set -- every correction call will fail and the "
                "session will disable itself after "
                f"{self.max_consecutive_failures} failures (the run then behaves as a plain "
                "rollout with no interventions).",
                flush=True,
            )

        # Live policy version for a pooled run, stamped onto every sample (see cast_relabel).
        self.policy_version = 0

        # ── mutable run state ──────────────────────────────────────────────────────────
        self.episode = 0
        self.route = ""
        self._episode_step = 0
        self._global_step = -1
        self._cot_queries = 0
        self._consecutive_failures = 0
        self._disabled = False
        self._lock = threading.Lock()
        # One-worker pool for the bounded coach call (see _call_coach_bounded), plus the future of
        # a call that timed out and is still draining in the background.
        self._executor: Any = None
        self._inflight: Any = None

        # Rolling buffer of stride-aligned recent model inputs, long enough to cover the lead-up.
        buf_len = max(1, self.pre_intervention_steps // self.pre_intervention_stride_steps + 1)
        self._recent: deque[dict[str, Any]] = deque(maxlen=buf_len)
        self._ego_hist: deque[tuple[float, float]] = deque(maxlen=self.ego_history_len)
        self._subtask_history: deque[str] = deque(maxlen=max(1, self.subtask_history_len))
        # Correction decided by the actor for an env step whose action does not exist yet.
        self._pending: dict[str, Any] | None = None
        # Sample dirs written this episode, so finalize_episode can patch their outcome tag.
        self._episode_sample_dirs: list[Path] = []
        self._kept_seen = 0

        # ── counters (surfaced to wandb by main_carla) ─────────────────────────────────
        self.num_queries = 0
        self.num_interventions = 0
        self.num_failures = 0
        self.num_samples_written = 0
        self.last_query_sec = 0.0
        self._last_query_wall = 0.0

    # ── episode lifecycle ──────────────────────────────────────────────────────────────
    def begin_episode(
        self,
        *,
        episode_count: int = 0,
        route_id: str = "",
        route_name: str = "",
        **_: Any,
    ) -> None:
        """Signature mirrors ``OnlineCastRelabelSession.begin_episode`` so the call sites match.

        ``route_id`` is the scenario (what a merged corpus is split by); ``route_name`` is the
        current routing *command*, which is scene context and is only used in the prompt header
        when no scenario is known.
        """
        self.episode = int(episode_count)
        self.route = str(route_id or route_name or "")

    def reset_episode(self) -> None:
        self._recent.clear()
        self._ego_hist.clear()
        self._subtask_history.clear()
        self._pending = None
        self._episode_sample_dirs = []
        self._episode_step = 0

    def set_step(self, *, episode_step: int, global_step: int) -> None:
        """Stamp the env step the actor is about to act on.

        Called from the rollout loop immediately before action selection, so the correction the
        actor decides inside :meth:`cot_intervention` can be attributed to the right step without
        the actor having to know anything about episode bookkeeping.
        """
        self._episode_step = int(episode_step)
        self._global_step = int(global_step)

    # ── actor-side: the intervention itself ────────────────────────────────────────────
    def cot_intervention(self, query: dict[str, Any]) -> tuple[str, str] | None:
        """``SteerVLAActor.cot_intervention_fn``: correct a freshly sampled CoT, or keep it.

        ``query`` carries the decoded ``subtask`` / ``reasoning`` plus the scene the actor built
        them from (``image`` / ``state`` / ``current_speed`` / ``prompt`` / ``routing_command``).
        Returns ``(subtask, reasoning)`` to override the conditioning, or ``None`` to leave the
        model's own CoT in place.

        Never raises: this sits on the driving path, so every failure mode -- no API key, a
        timeout, a malformed response -- degrades to ``None`` (drive on the model's own language).
        """
        try:
            return self._cot_intervention_inner(query)
        except Exception as exc:  # noqa: BLE001 - the rollout must survive any coach failure.
            self._note_failure(exc)
            return None

    def _cot_intervention_inner(self, query: dict[str, Any]) -> tuple[str, str] | None:
        subtask = strip_cot_sentinels(query.get("subtask"))
        reasoning = strip_cot_sentinels(query.get("reasoning"))
        if subtask:
            self._subtask_history.append(subtask)
        if self._disabled or not self._should_query():
            self._park_kept(query, subtask, reasoning)
            return None

        image = query.get("image")
        if image is None:
            self._park_kept(query, subtask, reasoning)
            return None

        prompt = build_foresight_correction_prompt(
            subtask=subtask,
            reasoning=reasoning,
            routing_command=str(query.get("routing_command") or ""),
            telemetry=self._telemetry(query),
            subtask_history=list(self._subtask_history)[:-1] if self.subtask_history_len else None,
            memory_block=(self._memory.render() if self._memory is not None else ""),
            seed_subtasks=self.seed_subtasks,
        )

        t0 = time.time()
        response = self._call_coach_bounded(np.asarray(image), prompt)
        self.last_query_sec = time.time() - t0
        self._last_query_wall = time.time()
        if response is None:  # still in flight from a previous step; drive on unchanged.
            self._park_kept(query, subtask, reasoning)
            return None
        self.num_queries += 1
        self._consecutive_failures = 0

        correction = parse_foresight_correction(response)
        if self.save_artifacts:
            self._write_query_artifact(query, subtask, reasoning, prompt, correction)

        if not correction.is_intervention:
            self._park_kept(query, subtask, reasoning)
            return None

        self.num_interventions += 1
        if self._memory is not None:
            # Same cross-call ledger cast_relabel keeps, so successive frames do not undo each
            # other ("remain stopped -> accelerate: 7x"). ``CorrectionMemory`` is written against
            # the CAST window schema, so one correction is fed to it as a one-chunk window; the
            # env step stands in for the window index, which is what the rendered notes show
            # ("w1234: 1x remain stopped -> accelerate"). It records only *longitudinal* intent
            # changes, so a purely lateral correction is (deliberately) not remembered.
            try:
                self._memory.observe_window(
                    {
                        "action_chunks": [
                            {
                                "label": LABEL_INTERVENTION,
                                "original_subtask": subtask,
                                "suggested_subtasks": [correction.subtask],
                            }
                        ]
                    },
                    window_index=int(self._episode_step),
                    route=self.route,
                )
            except Exception:  # noqa: BLE001 - memory is an aid, never a hard dependency.
                pass
        # Overwrite the history entry: what the vehicle is about to do is the corrected subtask,
        # not the one the model proposed, so the next query's continuity context must show that.
        if self._subtask_history:
            self._subtask_history[-1] = correction.subtask
        self._park_correction(query, subtask, reasoning, correction)
        return correction.subtask, correction.reasoning

    def _call_coach_bounded(self, image: np.ndarray, prompt: str) -> str | None:
        """``complete_image_text`` with a hard wall-clock bound.

        ``vlm_feedback._gemini_generate_content`` uses a 120 s socket timeout and retries 429/5xx
        up to five times with exponential backoff -- worst case well over two minutes inside one
        call. That is fine for a hindsight window review, but this call sits on the driving path
        with the CARLA sim paused around it, so it needs its own ceiling.

        The HTTP call cannot be cancelled once started, so a timed-out request is *abandoned*
        rather than killed: the worker keeps draining it in the background and the next few CoT
        queries return ``None`` (drive on the policy's own CoT) until it finishes. That is
        deliberately preferred to queueing corrections that are already stale by the time they
        arrive -- a correction is only useful for the frame it was asked about.
        """
        import concurrent.futures

        if self._inflight is not None:
            if not self._inflight.done():
                return None
            self._inflight = None
        if self._executor is None:
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="yay-robot-coach"
            )
        future = self._executor.submit(self._coach.complete_image_text, image, prompt)
        try:
            return future.result(timeout=max(1.0, self.request_timeout_sec))
        except concurrent.futures.TimeoutError:
            self._inflight = future
            raise TimeoutError(
                f"coach did not answer within {self.request_timeout_sec:.0f}s; abandoning the call"
            ) from None

    def close(self) -> None:
        """Release the coach worker thread. Safe to call more than once."""
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
        self._inflight = None

    def _should_query(self) -> bool:
        self._cot_queries += 1
        if (self._cot_queries - 1) % self.query_every_n_cot_queries != 0:
            return False
        return not (
            self.min_seconds_between_queries > 0.0
            and self._last_query_wall > 0.0
            and time.time() - self._last_query_wall < self.min_seconds_between_queries
        )

    def _telemetry(self, query: dict[str, Any]) -> dict[str, Any]:
        state = query.get("state")
        speed = float(query.get("current_speed", 0.0) or 0.0)
        yaw_rate = 0.0
        if state is not None:
            flat = np.asarray(state, dtype=np.float32).reshape(-1)
            if flat.size > EGO_STATE_IDX_SPEED:
                speed = float(flat[EGO_STATE_IDX_SPEED])
            if flat.size > EGO_STATE_IDX_YAW_RATE:
                yaw_rate = float(flat[EGO_STATE_IDX_YAW_RATE])
        return {
            "route": self.route,
            "episode": int(self.episode),
            "episode_step": int(self._episode_step),
            "current_speed_mps": round(speed, 2),
            "yaw_rate_deg_s": round(yaw_rate, 2),
            "stopped": bool(speed < 0.2),
            "interventions_so_far": int(self.num_interventions),
        }

    def _note_failure(self, exc: Exception) -> None:
        self.num_failures += 1
        self._consecutive_failures += 1
        print(f"[yay_robot] correction call failed ({type(exc).__name__}: {exc})", flush=True)
        if (
            self.max_consecutive_failures > 0
            and self._consecutive_failures >= self.max_consecutive_failures
            and not self._disabled
        ):
            self._disabled = True
            print(
                f"[yay_robot] {self._consecutive_failures} consecutive failures -- disabling "
                "corrections for the rest of the run; the rollout continues on the policy's own "
                "CoT and no further intervention samples are collected.",
                flush=True,
            )

    # ── parking a decision until the executed action exists ────────────────────────────
    def _park_correction(
        self,
        query: dict[str, Any],
        subtask: str,
        reasoning: str,
        correction: CotCorrection,
    ) -> None:
        with self._lock:
            self._pending = {
                "kind": "intervention",
                "episode_step": int(self._episode_step),
                "global_step": int(self._global_step),
                "scene": self._scene_from_query(query),
                "original_subtask": subtask,
                "original_reasoning": reasoning,
                "subtask": correction.subtask,
                "reasoning": correction.reasoning,
                "rationale": correction.rationale,
            }

    def _park_kept(self, query: dict[str, Any], subtask: str, reasoning: str) -> None:
        if not (self.store_hl_dataset and self.store_kept_samples) or not subtask:
            return
        self._kept_seen += 1
        if (self._kept_seen - 1) % self.kept_sample_stride != 0:
            return
        with self._lock:
            self._pending = {
                "kind": "kept",
                "episode_step": int(self._episode_step),
                "global_step": int(self._global_step),
                "scene": self._scene_from_query(query),
                "original_subtask": subtask,
                "original_reasoning": reasoning,
                "subtask": subtask,
                "reasoning": reasoning,
                "rationale": "",
            }

    @staticmethod
    def _scene_from_query(query: dict[str, Any]) -> dict[str, Any]:
        image = query.get("image")
        state = query.get("state")
        return {
            "image": None if image is None else np.asarray(image, dtype=np.uint8),
            "state": None if state is None else np.asarray(state, dtype=np.float32).reshape(-1),
            "current_speed": float(query.get("current_speed", 0.0) or 0.0),
            "prompt": str(query.get("prompt") or ""),
            "routing_command": str(query.get("routing_command") or ""),
        }

    # ── loop-side: buffer frames, flush pending decisions ──────────────────────────────
    def record_model_input(
        self,
        *,
        episode_step: int,
        image: np.ndarray | None,
        state: np.ndarray | None,
        current_speed: float = 0.0,
        prompt: str = "",
        subtask: str = "",
        reasoning: str = "",
        action_chunk: np.ndarray | None = None,
        routing_command: str = "",
        global_step: int = -1,
    ) -> None:
        """Buffer this env step's model input and flush any decision parked for it.

        Signature deliberately matches ``OnlineCastRelabelSession.record_model_input`` so the two
        sessions are interchangeable at the call site in ``main_carla``. Called every env step
        with the *pre-step* obs the action was taken from; only stride-aligned steps are retained
        as lead-up candidates.
        """
        self._push_ego_history(state, current_speed)
        entry = {
            "episode_step": int(episode_step),
            "global_step": int(global_step),
            "image": None if image is None else np.asarray(image, dtype=np.uint8),
            "state": None if state is None else np.asarray(state, dtype=np.float32).reshape(-1),
            "ego_hist": self._ego_history_array(),
            "current_speed": float(current_speed),
            "prompt": str(prompt or ""),
            "routing_command": str(routing_command or ""),
            "subtask": str(subtask or ""),
            "reasoning": str(reasoning or ""),
            "action_chunk": None if action_chunk is None else np.asarray(action_chunk, dtype=np.float32),
            "policy_version": int(self.policy_version),
        }

        with self._lock:
            pending = self._pending
            self._pending = None
        if pending is not None and self.store_hl_dataset:
            self._flush_pending(pending, entry)

        # Buffer *after* the flush, so an intervention's own frame is not also offered to itself
        # as a lead-up frame.
        if image is not None and (int(episode_step) - 1) % self.pre_intervention_stride_steps == 0:
            self._recent.append(entry)

    def _push_ego_history(self, state: np.ndarray | None, current_speed: float) -> None:
        """Append this step's ``[speed_mps, course_deg]`` pair (see cast_relabel for the layout)."""
        speed = float(current_speed)
        course = 0.0
        if state is not None:
            flat = np.asarray(state, dtype=np.float32).reshape(-1)
            if flat.size > EGO_STATE_IDX_SPEED:
                speed = float(flat[EGO_STATE_IDX_SPEED])
            if flat.size > EGO_STATE_IDX_YAW_RATE:
                course = float(flat[EGO_STATE_IDX_YAW_RATE]) * SIMLINGO_FRAME_DT
        self._ego_hist.append((speed, course))

    def _ego_history_array(self) -> np.ndarray:
        hist = list(self._ego_hist)
        if not hist:
            return np.zeros((self.ego_history_len, 2), dtype=np.float32)
        while len(hist) < self.ego_history_len:
            hist.insert(0, hist[0])
        return np.asarray(hist[-self.ego_history_len :], dtype=np.float32)

    def _flush_pending(self, pending: dict[str, Any], entry: dict[str, Any]) -> None:
        """Turn a parked decision (+ the now-known executed action) into HL samples on disk."""
        kind = str(pending.get("kind"))
        # The recorded step is authoritative, not the stamp the actor parked: a decision is always
        # flushed by the very next ``record_model_input``, and taking the stamp would mis-place the
        # lead-up window if a CoT were ever queried before the loop's first ``set_step``.
        step = int(entry["episode_step"])
        subtask = str(pending.get("subtask") or "")
        reasoning = str(pending.get("reasoning") or "")
        if not subtask or not reasoning:
            return

        samples: list[HLSample] = []
        if kind == "kept":
            samples.append(
                self._build_sample(
                    entry,
                    subtask=subtask,
                    reasoning=reasoning,
                    original_subtask=str(pending.get("original_subtask") or ""),
                    original_reasoning=str(pending.get("original_reasoning") or ""),
                    label=LABEL_KEPT,
                    credit_source="",
                    # The model's own CoT produced this action, so the pair is a valid
                    # action-supervision target if a downstream converter ever wants one.
                    action_matches_subtask=True,
                    chunk_index=0,
                )
            )
            tag = f"ep{self.episode:04d}_step{step:06d}_kept"
        else:
            # The intervention frame: the flow expert was conditioned on the CORRECTED language
            # for this very step, so the executed action does match the stored subtask -- which is
            # exactly what hindsight relabeling cannot say about its own samples.
            samples.append(
                self._build_sample(
                    entry,
                    subtask=subtask,
                    reasoning=reasoning,
                    original_subtask=str(pending.get("original_subtask") or ""),
                    original_reasoning=str(pending.get("original_reasoning") or ""),
                    label=LABEL_INTERVENTION,
                    credit_source=CREDIT_DIRECT,
                    action_matches_subtask=True,
                    chunk_index=0,
                )
            )
            samples.extend(self._lead_up_samples(step, subtask, reasoning))
            tag = f"ep{self.episode:04d}_step{step:06d}_intervention"

        out_dir = self.hl_dataset_dir / tag
        try:
            write_hl_samples(samples, out_dir)
        except Exception as exc:  # noqa: BLE001 - a disk hiccup must not end the rollout.
            print(f"[yay_robot] failed to write HL samples to {out_dir}: {exc}", flush=True)
            return
        self._episode_sample_dirs.append(out_dir)
        self.num_samples_written += len(samples)
        if self.debug and kind != "kept":
            print(
                f"[yay_robot] intervention @ ep{self.episode} step {step}: "
                f"{pending.get('original_subtask')!r} -> {subtask!r} "
                f"({len(samples)} samples incl. {len(samples) - 1} lead-up)",
                flush=True,
            )

    def _lead_up_samples(self, step: int, subtask: str, reasoning: str) -> list[HLSample]:
        """The buffered frames inside ``pre_intervention_sec`` before ``step``, corrected-labeled.

        Newest first so the cap keeps the frames closest to the intervention -- those are the ones
        where the corrected subtask is most clearly the right call from what is already visible.
        """
        if self.pre_intervention_steps <= 0:
            return []
        earliest = step - self.pre_intervention_steps
        candidates = [
            e for e in self._recent if earliest <= int(e["episode_step"]) < step and e["image"] is not None
        ]
        candidates.sort(key=lambda e: int(e["episode_step"]), reverse=True)
        if self.max_pre_intervention_samples > 0:
            candidates = candidates[: self.max_pre_intervention_samples]
        return [
            self._build_sample(
                e,
                subtask=subtask,
                reasoning=reasoning,
                original_subtask=str(e.get("subtask") or ""),
                original_reasoning=str(e.get("reasoning") or ""),
                label=LABEL_INTERVENTION,
                credit_source=CREDIT_PRECURSOR,
                # These actions were driven on the UNcorrected language, so the stored chunk is
                # not a target for the corrected subtask.
                action_matches_subtask=False,
                chunk_index=int(e["episode_step"]) - step,
            )
            for e in candidates
        ]

    def _build_sample(
        self,
        entry: dict[str, Any],
        *,
        subtask: str,
        reasoning: str,
        original_subtask: str,
        original_reasoning: str,
        label: str,
        credit_source: str,
        action_matches_subtask: bool,
        chunk_index: int,
    ) -> HLSample:
        state_vec = entry.get("state")
        if state_vec is None:
            state_vec = np.zeros((1,), dtype=np.float32)
        ego_hist = entry.get("ego_hist")
        if ego_hist is None:
            ego_hist = _ego_hist_from_state(
                state_vec,
                current_speed=float(entry.get("current_speed", 0.0)),
                ego_history_len=self.ego_history_len,
            )
        return HLSample(
            image=np.asarray(entry["image"], dtype=np.uint8),
            state=np.asarray(state_vec, dtype=np.float32).reshape(-1),
            current_speed=float(entry.get("current_speed", 0.0)),
            prompt=str(entry.get("prompt") or ""),
            subtask=subtask,
            reasoning=reasoning,
            actions=_shape_hl_action_chunk(
                entry.get("action_chunk"), self.action_chunk_steps, self.hl_action_dim
            ),
            # HL-only supervision, exactly as cast_relabel writes it: the CoT/VLM backbone is
            # trained, the action expert is not.
            action_loss_mask=np.zeros((int(self.action_chunk_steps),), dtype=bool),
            episode=int(self.episode),
            # No windows in this method; ``window_index`` is repurposed as the intervention's env
            # step so provenance survives into the manifest without a schema change.
            window_index=int(entry.get("episode_step", -1)),
            chunk_index=int(chunk_index),
            episode_step=int(entry.get("episode_step", -1)),
            label=label,
            credit_source=credit_source,
            action_matches_subtask=bool(action_matches_subtask),
            ego_hist=np.asarray(ego_hist, dtype=np.float32),
            routing_command=str(entry.get("routing_command") or ""),
            original_subtask=original_subtask,
            original_reasoning=original_reasoning,
            route=self.route,
            global_step=int(entry.get("global_step", -1)),
            policy_version=int(entry.get("policy_version", self.policy_version)),
            # Backfilled by finalize_episode once the episode's fate is known.
            outcome="",
        )

    # ── artifacts + episode close-out ──────────────────────────────────────────────────
    def _write_query_artifact(
        self,
        query: dict[str, Any],
        subtask: str,
        reasoning: str,
        prompt: str,
        correction: CotCorrection,
    ) -> None:
        """Append this query to ``yay_robot/queries.jsonl`` (prompt + raw response + verdict)."""
        record = {
            "episode": int(self.episode),
            "episode_step": int(self._episode_step),
            "global_step": int(self._global_step),
            "route": self.route,
            "original_subtask": subtask,
            "original_reasoning": reasoning,
            "verdict": "CORRECT" if correction.is_intervention else "KEEP",
            "corrected_subtask": correction.subtask,
            "corrected_reasoning": correction.reasoning,
            "rationale": correction.rationale,
            "query_seconds": round(self.last_query_sec, 3),
            "prompt": prompt,
            "response": correction.raw_response,
        }
        try:
            with (self.artifact_dir / "queries.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except Exception:  # noqa: BLE001 - artifacts are never worth failing a rollout over.
            pass

    def finalize_episode(self, *, metadata: dict[str, Any] | None = None) -> str:
        """Stamp the episode's outcome onto every manifest written during it.

        Samples are written as they are produced (the run trains on them the same episode), so the
        ``outcome`` field that ``SteerVLAActor``'s ``use_adaptive_sampling`` reads cannot be filled
        in at write time. This patches it in afterwards -- JSON only, the ``.npz`` payloads are
        untouched. Samples written before this ran simply weigh neutrally in the meantime.
        """
        outcome = resolve_window_outcome(dict(metadata or {}))
        if outcome:
            for out_dir in self._episode_sample_dirs:
                manifest_path = out_dir / "hl_samples.json"
                try:
                    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                    for sample in payload.get("samples", []):
                        sample["outcome"] = outcome
                    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                except Exception:  # noqa: BLE001 - a missing/partial manifest is not fatal.
                    continue
        self._episode_sample_dirs = []
        return outcome

    def wandb_metrics(self) -> dict[str, float]:
        """Counters for the per-step W&B log line in ``main_carla``."""
        rate = self.num_interventions / self.num_queries if self.num_queries else 0.0
        return {
            "yay_robot/queries": float(self.num_queries),
            "yay_robot/interventions": float(self.num_interventions),
            "yay_robot/intervention_rate": float(rate),
            "yay_robot/failures": float(self.num_failures),
            "yay_robot/samples_written": float(self.num_samples_written),
            "yay_robot/last_query_sec": float(self.last_query_sec),
            "yay_robot/disabled": float(self._disabled),
        }
