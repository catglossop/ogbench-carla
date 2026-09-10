"""Standalone checks for run reproducibility: seed splitting, eval reseeding, run summary.

Run directly (no model, no CARLA, no VLM)::

    JAX_PLATFORMS=cpu PYTHONPATH=impls uv run python impls/test_run_determinism.py

Covers the 2026-09-09 change: --carla_seed and --train_seed are separate, the model's sampling is
actually a function of the train seed (it used to be a function of the call counter alone), and
--eval-mode turns on everything a reportable run needs -- pinned simulator seed, 2000-step
checkpoints, a final weights export, eval episodes that replay the training carla seed while
varying only the model seed, and run_summary.json. None of that is enforced without the flag.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

FAILURES: list[str] = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  -- ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


# ── 1. seed resolution ────────────────────────────────────────────────────────────────
print("\n[1] resolve_run_seeds")
import main_carla
from main_carla import FLAGS, resolve_run_seeds, write_run_summary

FLAGS([sys.argv[0]])  # parse defaults so FLAGS is usable

FLAGS.seed, FLAGS.carla_seed, FLAGS.train_seed = 7, -1, -1
FLAGS.eval_seeds, FLAGS.post_stop_eval_episodes = "", 3
c, t, e = resolve_run_seeds()
check("both seeds fall back to --seed", (c, t) == (7, 7), f"got {(c, t)}")
check("eval seeds derived, one per eval episode", e == [1008, 1009, 1010], f"got {e}")
check("eval seeds differ from the train seed", t not in e)

FLAGS.carla_seed, FLAGS.train_seed = 11, 22
c, t, e = resolve_run_seeds()
check("explicit seeds win over --seed", (c, t) == (11, 22), f"got {(c, t)}")

FLAGS.eval_seeds = "1,2,3"
_, _, e = resolve_run_seeds()
check("comma-separated eval seeds parse", e == [1, 2, 3], f"got {e}")
FLAGS.eval_seeds = "4 5"
_, _, e = resolve_run_seeds()
check("whitespace-separated eval seeds parse", e == [4, 5], f"got {e}")
FLAGS.eval_seeds = ""

# ── 2. the checkpoint cadence is a real constant, not a config default ─────────────────
print("\n[2] online checkpoint cadence")
check(
    "DEFAULT_ONLINE_CKPT_EVERY_STEPS is 2000",
    main_carla.DEFAULT_ONLINE_CKPT_EVERY_STEPS == 2000,
    f"got {main_carla.DEFAULT_ONLINE_CKPT_EVERY_STEPS}",
)
src = Path("impls/main_carla.py").read_text()
check(
    "an unset interval is corrected when updates are on",
    "_hl_ckpt_every = DEFAULT_ONLINE_CKPT_EVERY_STEPS" in src,
)
check(
    "the weights training produced are exported at the stop",
    "_save_steervla_ckpt(int(step), final=True)" in src,
)
check(
    "periodic checkpointing stops once the eval phase begins",
    "        if not eval_phase:\n            if agent is not None and any_updates_on" in src
    and "            _save_steervla_ckpt(step)" in src,
)
# The end-of-training export must be the LAST checkpoint written: eval-phase saves are
# byte-identical copies and, under hl_checkpoint_keep_last, rotate the real one off disk.
_stop_at = src.index("final_train_driving_score = _ep_ds")
_periodic_at = src.index("        if not eval_phase:\n            if agent is not None")
check(
    "the final export happens before the gated periodic save in the same iteration",
    _stop_at < _periodic_at,
)

# ── 2b. ... but only under --eval-mode ────────────────────────────────────────────────
print("\n[2b] everything is gated behind --eval_mode")
check("the flag exists and defaults off", FLAGS.eval_mode is False)
check(
    "checkpoint-interval correction is gated",
    "if FLAGS.eval_mode and _hl_updating and _hl_ckpt_every <= 0:" in src,
)
check("--save_interval tightening is gated", "        FLAGS.eval_mode\n        and any_updates_on" in src)
_final_export = src[src.index("final_train_driving_score = _ep_ds"):]
check("final weights export is gated",
      _final_export.index("if FLAGS.eval_mode:") < _final_export.index("_save_steervla_ckpt(int(step), final=True)"))
check("eval reseeding is gated", "if eval_phase and FLAGS.eval_mode:" in src)
check("run_summary.json is gated", "                    if FLAGS.eval_mode:\n                        write_run_summary(" in src)
check("simulator-seed pin is gated", "if FLAGS.eval_mode or FLAGS.eval_only:" in src)
check(
    "the frozen-eval local was renamed so it cannot be confused with the flag",
    "eval_phase = False" in src and "    eval_mode = False" not in src,
)

# ── 2c. run_carla.sh exposes it ───────────────────────────────────────────────────────
print("\n[2c] run_carla.sh --eval-mode")
sh = Path("run_carla.sh").read_text()
check("default is off", 'EVAL_MODE="false"' in sh)
check("flag is parsed (both spellings)", "--eval-mode|--eval_mode)" in sh)
check("bare --eval-mode means true", 'else EVAL_MODE="true"; shift; fi ;;' in sh)
check("explicit true/false is honoured", 'if [[ "${2:-}" == "true" || "${2:-}" == "false" ]]' in sh)
check("forwarded to main_carla", '--eval_mode="${EVAL_MODE}"' in sh)
check("documented in --help", "--eval-mode [true|false]" in sh)
check("--carla-seed exposed", "--carla-seed|--carla_seed)" in sh)
check("--train-seed exposed", "--train-seed|--train_seed)" in sh)
check("--eval-seeds exposed", "--eval-seeds|--eval_seeds)" in sh)
check(
    "seeds forwarded only when set, so the --seed fallback still applies",
    '${CARLA_SEED:+--carla_seed="${CARLA_SEED}"}' in sh
    and '${TRAIN_SEED:+--train_seed="${TRAIN_SEED}"}' in sh
    and '${EVAL_SEEDS:+--eval_seeds="${EVAL_SEEDS}"}' in sh,
)

# ── 3. eval replays the training carla seed, varying only the model ───────────────────
print("\n[3] eval reseeding")
check("eval episodes reset with the training carla seed", "_reset_seed = int(_carla_seed)\n" in src)
check("training episodes still walk the seed", "_reset_seed = int(_carla_seed) + episode_count" in src)
check(
    "each eval episode sets a different model sampling seed",
    "steervla_actor.sampling_seed = int(_eval_seeds[_eval_idx])" in src,
)
check("the simulator seed is pinned under --eval-mode (or --eval_only)",
      'extra_carla["traffic_manager_seed"] = int(_carla_seed_rc)' in src)

# ── 4. model sampling is a function of the seed, not the call counter ─────────────────
print("\n[4] SteerVLAActor sampling seed")
vsrc = Path("impls/vlas/steervla.py").read_text()
check("no sampling key is derived from the bare call counter",
      "jax.random.PRNGKey(self._call_counter)" not in vsrc)
check("a seeded helper exists", "def _next_sampling_rng(self)" in vsrc)
check("the seed is folded into every draw",
      "jax.random.fold_in(jax.random.PRNGKey(int(self.sampling_seed)), self._call_counter)" in vsrc)



class _Shim:
    def __init__(self, seed):
        self._call_counter = 0
        self.sampling_seed = seed

    _next_sampling_rng = None


# Bind the real method without importing the (heavy) actor module.
_ns = {}
exec(  # noqa: S102 - executing our own source under test, not user input
    "import jax\n"
    + vsrc[vsrc.index("    def _next_sampling_rng(self):"): vsrc.index("    def sample_candidates(")]
    .replace("    def", "def", 1).replace("\n        ", "\n    "),
    _ns,
)
_Shim._next_sampling_rng = _ns["_next_sampling_rng"]

a, b = _Shim(0), _Shim(1)
ka = [a._next_sampling_rng() for _ in range(3)]
kb = [b._next_sampling_rng() for _ in range(3)]
check("different seeds give different keys", not any(bool((x == y).all()) for x, y in zip(ka, kb)))
c1, c2 = _Shim(5), _Shim(5)
check(
    "the same seed reproduces the same key sequence",
    all(bool((x == y).all()) for x, y in zip(
        [c1._next_sampling_rng() for _ in range(3)],
        [c2._next_sampling_rng() for _ in range(3)])),
)
d = _Shim(0)
k1, k2 = d._next_sampling_rng(), d._next_sampling_rng()
check("successive calls decorrelate", not bool((k1 == k2).all()))

# ── 5. run_summary.json carries everything needed to reproduce and compare ────────────
print("\n[5] run_summary.json")
with tempfile.TemporaryDirectory() as td:
    out = write_run_summary(
        td, route="highway-exit-002", carla_seed=11, train_seed=22,
        eval_seeds=[1, 2, 3], eval_scores=[10.0, 20.0, 30.0],
        final_train_driving_score=44.5, stop_reason="150 HL updates applied >= cap 150",
        hl_updates_applied=150, env_steps=3100, final_checkpoint="/ckpt/3100",
    )
    written = json.loads((Path(td) / "run_summary.json").read_text())
check("file is written", written == out)
check("(a) driving score at end of training", written["training"]["final_driving_score"] == 44.5)
check("(b) train and carla seed", written["seeds"] == {"carla_seed": 11, "train_seed": 22})
check(
    "(c) each eval seed paired with its score",
    written["eval"] == [{"eval_seed": 1, "driving_score": 10.0},
                        {"eval_seed": 2, "driving_score": 20.0},
                        {"eval_seed": 3, "driving_score": 30.0}],
)
check("final checkpoint recorded", written["training"]["final_checkpoint"] == "/ckpt/3100")
check("eval mean computed", written["eval_mean_driving_score"] == 20.0)

print()
if FAILURES:
    print(f"FAILED ({len(FAILURES)}): " + ", ".join(FAILURES))
    raise SystemExit(1)
print("all checks passed")
