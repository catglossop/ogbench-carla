# OGBench-CARLA research and engineering handoff

This document is the durable context for continuing CARLA VLA / residual-RL work
on another machine. It deliberately excludes credentials and API keys.

## Current state

- Working branch: `dev`.
- Local handoff commit: `daac58a add resumable residual sweep runner`.
- The commit was **not pushed** from the old host: `git push origin dev` failed
  because that host did not have GitHub credentials. Push it from an authenticated
  terminal, or transfer the commit/bundle directly.
- The old hung sweep process group was terminated. Its completion state is intact.
- The active sweep is `coarse_grid_v3`, defined in
  `impls/configs/residual_rl_sweeps.yaml` and executed by
  `run_residual_sweep.sh`.
- See the `Residual-RL sweep` section before changing a profile label: status and
  artifact namespaces depend on that label.

## Repository map

The repository primarily runs VLA experiments in the CARLA driving simulator.

- `run_carla.sh`: the normal entrypoint. It chooses CARLA assets/configuration,
  launches UE4 and Xvfb, starts Python, handles native-process retry/restart, and
  translates launcher flags into `main_carla.py` flags.
- `impls/main_carla.py`: the main online-training/evaluation driver. This is the
  most important file for rollout, replay, residual SAC, best-of-N, EXPO,
  checkpointing, video annotation, W&B logging, and route-specific execution.
- `impls/vlas/steervla.py`: SteerVLA actor loading/inference, action chunks,
  prompt/CoT processing, and related cached-query cadence.
- `impls/configs/`: agent configurations. `steervla_residual_config.py` is the
  normal residual-SAC base configuration.
- `impls/jax_agents/sac_residual.py`: residual-SAC agent and configuration;
  residual tether/magnitude regularization lives here.
- `ogbench/carla/carla_utils.py`: CARLA Bench2Drive wrapper and simulator lifecycle.
- `impls/carla_env_server.py`: CARLA 0.9.15 environment subprocess. The main
  Python process communicates with it via JSON lines because the JAX process and
  CARLA client use different Python/runtime requirements.
- `impls/coaches/`: CAST / HL-DAgger relabeling and coach logic.

The major experiment families are:

1. CAST relabel / HL DAgger: hindsight-capable VLM coach feedback is used to
   update the policy.
2. Best-of-N: sample candidate base-policy actions and select/evaluate them.
3. RL methods: residual RL (the present focus), GRPO, and related variants.

## CARLA process architecture and GPU mapping

A residual rollout is normally:

```text
run_residual_sweep.sh
  -> run_carla.sh
     -> main_carla.py (JAX / actor / critic)
        -> carla_env_server.py (CARLA 0.9.15 client)
           -> CarlaUE4 + Xvfb
```

GPU indexing is important:

- `CUDA_VISIBLE_DEVICES=<physical GPU>` makes the model see that physical GPU as
  logical CUDA device `0`; therefore `--train-gpu 0` is correct.
- CARLA's `--render-adapter` does **not** respect `CUDA_VISIBLE_DEVICES`; it must
  receive the physical GPU index.
- The sweep runner accepts one physical GPU and deliberately does:

```bash
CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" \
  ... --train-gpu 0 --render-adapter "$PHYSICAL_GPU"
```

Using simulator and model on the same physical GPU is intentional for this sweep.

## SteerVLA, prompt, and video annotation

Video annotation is built in `main_carla.py`. The video uses the native/front
camera observation and draws trajectory/action overlays plus a text panel.

Previously observed issues and conclusions:

- A camera view that showed more of the ego vehicle came from the source frame /
  camera path used for visualization, not from residual RL. It was a presentation
  change, not evidence of changed policy dynamics.
- Annotation was made unreadable/cropped because the bottom text panel did not
  robustly account for rendered text width/height and source-frame dimensions.
  The annotation fix is already present in earlier history (`0204f45 annotation
  fix`) and later `main_carla.py` work.
- Prompt display previously omitted the current-speed sentence, leaving only
  `follow the road`. The formatting now preserves the complete prompt/speed
  context before text wrapping/truncation.
- Verify any future annotation edits on actual recorded MP4 frames at the native
  output resolution. Check: top-left reward, all action fields, progress line,
  expert action, prompt, reasoning, subtask, panel bottom, and long strings.
  Do not rely only on a thumbnail/W&B player.

The current annotation panel deliberately truncates individual long fields rather
than allowing image width overflow. It should never crop the panel itself.

## Residual RL implementation

Residual RL keeps the frozen/base VLA action and learns a residual correction.
The final action is the base action plus the scaled residual (subject to the
normal action-space bounds). W&B commonly logs:

- `action/base_accel`, `action/base_steer`
- `action/residual_accel`, `action/residual_steer`
- `action/final_accel`, `action/final_steer`
- actor residual scale / absolute magnitude / loss / entropy / BC penalty
- rollout success, route progress, driving score, episode return

### Scale controls

The old single `residual_scale` is retained as a legacy fallback. The preferred
explicit controls are:

```text
agent.residual_accel_scale
agent.residual_steer_scale
```

Resolution rule in `main_carla.py`:

- an explicitly non-negative accel scale wins;
- an explicitly non-negative steer scale wins;
- `-1` means fall back (steer falls back to accel; accel falls back to legacy
  `residual_scale`).

This makes a steering scale of `-1` a tie-to-accel setting, not a literal negative
scale. Use `0.0` when steering residuals should be disabled.

The default residual state encoder is now `siglip_pool`; Pi-prefix is still
available as a comparison option.

### Tether / residual magnitude regularization

The ordinary residual magnitude tether already existed. The relevant sweep
parameter is `agent.residual_bc_beta`; it penalizes residual magnitude. The
current fixed-beta experiment uses:

```text
agent.residual_bc_normalize=false
```

This is intentional: a fixed beta should retain its direct scale rather than
being dynamically normalized. The low-level historical fallback had normalized
behavior enabled in some paths; config and agent defaults were aligned to avoid
an accidental discrepancy.

Only the ordinary magnitude tether is in scope for the current sweep. Do not add
other tether variants unless the experimental plan changes.

### Existing residual findings

A preliminary constrained-scale/magnitude-tether comparison was inconclusive:

- Constraints gave some apparent improvement in success / driving score, but the
  number of rollout points was sparse and noisy.
- In the `0.5` scale comparison, the constrained run did not necessarily show an
  action/residual EMA closer to zero. This is not a contradiction: the tether
  acts on the sampled/training residual magnitude (and can change the policy /
  critic/update distribution), whereas plotted rollout means are signed,
  state-distribution dependent averages. Positive/negative cancellation and
  changed visited states can conceal a magnitude reduction in a signed EMA.
- Earlier charts showed high-scale variants with residual absolute means around
  0.3 and low-scale variants around 0.06, confirming that the scale is active.

The best next evidence is the controlled coarse grid below, not reward tuning.

## EXPO / best-of-N conclusions

EXPO is currently out of scope for the sweep.

Observed failure mode:

- An EXPO run collapsed toward doing nothing: base/final accel drifted negative,
  residual accel stayed near zero, route progress and success fell to zero.
- EXPO critic values (`q_min`, `q_mean`, `q_max`) converged close to zero and
  candidate margins became tiny. Winner indices still varied, but the critic was
  not providing a useful ranking signal.
- The implementation's critic choice selected the best base candidate, then
  tested whether its associated residual improved it. This can miss a different
  base candidate whose *base + residual* has a superior value. This is a design
  limitation, not merely a visualization issue.

Gemini-backed EXPO selection was prototyped, including an uncertainty gate, but
was removed because Gemini was called too frequently and was too expensive/slow.
Do not re-enable it implicitly. If revisiting it, require an explicit budgeted
setup with both:

- an option to evaluate every decision; and
- a conservative uncertain-only gate based on a well-defined critic uncertainty
  measurement.

The GRPO implementation is the useful reference for external/VLM decision
plumbing. However, improving EXPO now should focus on a better critic (e.g., a
state/return head on SigLIP features, a proper joint candidate-residual critic,
or selectively queried Gemini), not reward tuning.

A conservative EXPO gate was discussed/implemented as an option: use residual
selection only in high-confidence/select instances, otherwise execute base.
It is unrelated to the current no-EXPO residual sweep.

## Critic notes

The residual critic has an ensemble/head array. The candidate values used by
EXPO are derived from the critic head outputs; understand the exact aggregation
(min/mean depending on the code path) before interpreting a single `q_*` chart.
The key caveat is that EXPO's former selection sequence is not a full joint
search over all `(base candidate, residual)` combinations.

A possible future critic experiment is to apply a Monte-Carlo-return-trained
state head on top of a SigLIP state encoder. This is conceptually modest wiring
if dimensions/preprocessing/checkpoint metadata match, but it requires explicit
validation of the encoder feature dimension, normalization, frozen/trainable
ownership, and state/action concatenation expected by the head. Do not assume a
Pi-prefix-trained head can consume SigLIP embeddings directly.

## Active residual-RL sweep

The sweep definition is versioned in `impls/configs/residual_rl_sweeps.yaml`.
`run_residual_sweep.sh` consumes it.

### Active profile: `coarse_grid_v3`

```yaml
routes:
  - enter-actor-flow-004
  - non-signalized-junction-right-turn-005
  - pedestrian-crossing-004
accel_scales: [0.1, 0.4]
steer_scales: [0.1, 0.4]
betas: [0.0, 5.0]
encoders: [siglip_pool]
seeds: [0]
online_steps: 20000
actions_per_model_query: 3
actions_per_cot: 5
expo: false
best_of_n: 1
otf_td_backup: false
residual_bc_normalize: false
```

This is a 24-job Cartesian grid (3 routes × 2 accel × 2 steer × 2 beta).
The left-turn route is intentionally excluded for now. `generalization-wall` is
also intentionally excluded: residual corrections are unlikely to be large
enough to repair its wall-crash failure.

The initial plan used `actions_per_model_query=3`, `actions_per_cot=3`. The user
corrected the intended CoT cadence to `5` after v2 had not actually been started.
Therefore the active/restarted profile is **v3**, preserving v2 for reproducible
identity but using CoT 5. Do not rename v3 back to v2.

The coarse v1 and initial-scale profiles are retained—not deleted—so historical
plans remain reproducible.

### Queue state at handoff

Output root:

```text
/raid/users/surya/carla_exps/residual_scale_sweep
```

Status directory:

```text
/raid/users/surya/carla_exps/residual_scale_sweep/status/coarse-grid-v3/
```

Completed markers (five):

1. `enter-actor-flow-004`, accel `0.1`, steer `0.1`, beta `0.0`
2. `enter-actor-flow-004`, accel `0.1`, steer `0.4`, beta `0.0`
3. `non-signalized-junction-right-turn-005`, accel `0.1`, steer `0.1`, beta `0.0`
4. `non-signalized-junction-right-turn-005`, accel `0.1`, steer `0.4`, beta `0.0`
5. `pedestrian-crossing-004`, accel `0.1`, steer `0.1`, beta `0.0`

The next job is:

```text
pedestrian-crossing-004__enc-siglip_pool__a-0.1__s-0.4__beta-0.0__seed-0
```

It failed before the first periodic resume checkpoint. Its run directory had
only `flags.json`, with no resume state, replay buffer, or model checkpoint.
Thus it must start at step 0; this is the only correct interpretation of
"resume" for that job.

### Running / continuing the queue

If the output root is shared between machines:

```bash
cd /path/to/ogbench-carla
git switch dev
git pull --ff-only origin dev
SWEEP_PROFILE=coarse_grid_v3 ./run_residual_sweep.sh <physical-gpu>
```

The script skips `.done` settings. It records a `.failed` marker for terminal
per-run failure and, by default, proceeds to the next setting. A later invocation
continues from remaining jobs. To retry a recorded failed setting:

```bash
RETRY_FAILED=1 SWEEP_PROFILE=coarse_grid_v3 ./run_residual_sweep.sh <physical-gpu>
```

An operator Ctrl-C exits the queue with code 130 and does not mark the currently
running job failed or silently advance. Use a new profile label for a deliberately
different matrix; retain a label when resuming the same matrix.

If the output root is not shared, transfer the status directory at minimum;
transfer logs/runs as well if retaining the previous artifacts locally:

```bash
rsync -a --info=progress2 \
  OLD_HOST:/raid/users/surya/carla_exps/residual_scale_sweep/ \
  /raid/users/surya/carla_exps/residual_scale_sweep/
```

The runner's W&B experiment names begin with
`residual-rl-hyperparam-sweep_` and include label, accel scale, steer scale,
beta, encoder, seed, and route. This avoids collisions with older experiments.

## CARLA crash / stall incident

The old host stalled while beginning job 6. Logs showed:

- UE4 launched and CARLA RPC initially became ready.
- Immediately after initialization, UE4/Xvfb became defunct.
- CARLA then tried to destroy sensor actors and reported a timeout of
  `7200000ms` while waiting for the simulator.
- The shell/python hierarchy was later found in stopped (`T`) state; continuing
  it exposed the same dead-server shutdown wait.

The root cause of the **queue hang** was not learning logic. It was teardown
against an already-dead CARLA server. The periodic recovery mechanism could not
help because the failure occurred before a recovery save.

Two guards now make this fail fast:

1. `ogbench/carla/carla_utils.py` checks the evaluator's UE4 subprocess state
   before RPC teardown and skips scenario/sensor destruction if UE4 is already
   dead.
2. `impls/carla_env_server.py` performs the same check before calling `env.close`
   when receiving shutdown, allowing the parent launcher to observe failure and
   apply its usual retry behavior.

The old process group was terminated after preserving queue markers. Do not
launch another queue against the same status root on the old host.

## Checkpoint/resume behavior

`main_carla.py` has periodic residual recovery (`resume_interval` defaults to
1000 steps). It saves model/replay/resume state under the run directory and
restores it when the launcher's `--resume=true` path is used.

Important distinction:

- `run_carla.sh` can retry a crashed run *only after the Python process exits*.
- A CARLA teardown RPC hang prevented that exit before the new guard.
- A job with no saved `resume_state` cannot continue within training; queue-level
  status markers only tell the sweep which parameter combinations are complete.

## Operational notes

- Actor checkpoints are commonly downloaded from cat-logs / GCP. The repository
  does not contain their credentials; authenticate separately on the new host.
- `GEMINI_API_KEY` may exist in the user environment, but do not log it, commit
  it, or make it a requirement for the residual sweep.
- CARLA assets can overwhelm VS Code source control. The local repository uses
  `.git/info/exclude` to hide untracked CARLA assets and similar local-only files.
  This is local configuration, not a committed `.gitignore` policy.
- Be careful with `ogbench/carla/`: local asset/source behavior has historically
  interacted with ignore rules. Verify `git status`, `git ls-files`, and the
  actual file content before assuming a CARLA wrapper edit will transfer.
- The working tree was clean immediately after `daac58a` was created.

## Recommended next research sequence

1. Finish `coarse_grid_v3` with at least the current single seed. Inspect
   success, progress, driving score, collisions, residual absolute magnitude,
   and actor BC/tether statistics together; do not select by signed residual EMA.
2. If a setting is promising, repeat it with more seeds before making a fine-grid
   decision.
3. Sweep higher acceleration scales only in the tether stage if the ordinary
   residual scale grid indicates corrections are too constrained.
4. Compare SigLIP and Pi-prefix only after establishing a reasonable scale/beta
   region. SigLIP is the default/baseline for current work.
5. Revisit EXPO only with an improved critic or a deliberately budgeted external
   selector. Do not tune reward merely to address the former EXPO collapse.

## Transfer options when GitHub push is unavailable

Preferred: authenticate and push the existing commit from the old host:

```bash
git push origin dev
```

Alternative: create a Git bundle on the old host and copy it to the new host:

```bash
# old host
git bundle create residual-sweep-handoff.bundle daac58a^..daac58a

# new host, from the repository
 git fetch /path/to/residual-sweep-handoff.bundle daac58a
 git cherry-pick daac58a
```

A chat session's hidden context does not transfer automatically. This document,
the commit, and the output-root status directory are the authoritative handoff.
