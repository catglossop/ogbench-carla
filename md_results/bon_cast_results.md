# Best-of-N + CAST Relabel — Results

**Date:** 2026-08-03 (final status 2026-08-07)
**Config:** `impls/configs/steervla_bon_cast_relabel_config.py` — **now deprecated**
**Critic checkpoint:** `/raid/users/cglossop/critic_ckpts/step_0012000.pkl`

> ## ⚠️ Superseded
>
> This config has since been deprecated in favour of **`steervla_bon_cast_config.py`**.
> Reason (per the deprecation note): the `BestOfNAgent` → `SteerVLAActor.sample_candidates`
> path draws all N CoTs in **one batched forward and never resets the actor's action cache
> between draws**, so candidates collapse toward the same action. Runs on this path gave poor
> results (W&B `yse2tuzl`, `w3lakhnw`). The replacement uses the flag-driven path
> (`--bon_critic_ckpt` → `main_carla._sample_diverse_candidates`), which samples one candidate
> at a time, resets the cache between draws, and resamples up to `--bon_max_sample_attempts`
> times per slot. It is also the exact base of `steervla_bon_cast_residual_config`, so the
> with/without-residual comparison is controlled.
>
> **Everything below describes the deprecated path.** The infrastructure findings (§GPU/hang/
> checkpoint/parser) still apply to any run of this stack; the *results* should not be used.
>
> ### The collapse was visible in this run's own data — and I under-weighted it
>
> Measured across both completed runs (candidates that share identical subtask text):
>
> | Run | Steps | Mean unique subtasks per 10 candidates | Steps where all 10 identical |
> |---|---:|---:|---:|
> | `enter-actor-flow-004` | 19,998 | 6.35 | 561 (2.8%) |
> | `signalized-junction-left-turn-001` | 19,998 | 7.08 | 203 (1.0%) |
>
> So ~3–4 of every 10 candidates were textual duplicates, at temperature 0.5. I reported
> candidate diversity as healthy on the basis of the **winner-index histogram** spreading across
> all 10 slots — but index spread does not imply *action* diversity, and a duplicated subtask
> can still win at a different index. I also logged `action_entropy_mean ≈ −2.66` without
> interpreting it; that low value was direct evidence of chunk collapse. The right check was
> unique-subtask count per step, which I did not run at the time.

---

## TL;DR

The two pipelines **do** merge through config alone, with one code fix each on the
supervision side and the selection side. The CAST-relabel half is verified working
end-to-end and produced 435 labeled action chunks feeding real OpenPI gradient steps.

**The Best-of-N half did not run in the first set of runs.** A gate in `main_carla.py`
silently disabled candidate selection whenever RL updates were off — which is exactly our
configuration, since the critic is frozen at a pretrained checkpoint. All numbers in the
"CAST relabel" section below are therefore from a **plain VLA rollout**, not a Best-of-N
rollout. Fixed and relaunched; see [Status](#status).

---

## 1. Does it merge through configs?

Yes. The two pipelines touch disjoint machinery:

| | Owns | Driven by |
|---|---|---|
| Best-of-N | action **selection** | `BestOfNAgent.sample_actions_with_vla` → `SteerVLAActor.sample_candidates` |
| CAST relabel | **supervision** | `OnlineCastRelabelSession`, built purely off `agent_config.cast_relabel` in `main_carla.py:1793` |

The seam is meaningful rather than coincidental: BoN stashes the *winning* candidate's CoT
via `stash_candidate_cot`, and CAST reads exactly that stash (`main_carla.py:3411`,
`record_model_input(subtask=..., reasoning=...)`). So every HL sample is keyed to the
critic-selected candidate. With `store_good_chunks=True`, GOOD chunks reinforce the subtask
the critic picked; BAD chunks replace it with the VLM's correction.

### Critic checkpoint compatibility

`step_0012000.pkl` is a DSRL-shaped checkpoint; only `modules_critic` is extracted.

```
modules_critic/value_net/Dense_0/kernel  (2, 3496, 256)
```

`3496 = 3 × 1152 + 40` — SigLIP image + prompt + subtask slots
(`siglip_include_prompt_subtask=True`), plus the 10×4 action chunk. Tree structure and every
leaf shape validated against `BestOfNAgent`'s critic at load. Confirmed in the log:

```
[best_of_n] loaded pretrained critic from /raid/users/cglossop/critic_ckpts/step_0012000.pkl
```

### One config consequence worth knowing

Setting `critic_pretrained_weights` is a **hard override**: `resolve_critic_feedback_mode`
(`coaches/critic_feedback.py:44`) forces mode `"none"`, and `BestOfNAgent.create` mirrors it
by dropping the SigLIP encoder and setting `lang_dim=0`. So candidate ranking is

```
argmax_i Q(obs_e, action_i)          # unconditioned on candidate subtask
```

not the subtask-conditioned `argmax_i Q([obs_e; subtask_i], action_i)` the Best-of-N docstring
describes. To get the conditioned variant, drop `critic_pretrained_weights` and train the
critic from scratch.

---

## 2. Code fixes required

Two real gaps; neither was reachable through config.

### 2.1 `BestOfNAgent.update_with_vla` ignored the HL update

`impls/jax_agents/best_of_n.py:694`. The method took `run_hl` "for signature parity" as a
no-op, and did not accept `global_step` — which `main_carla.py:3725` always passes. So
enabling any update with `agent_name="best_of_n"` raised `TypeError` before doing anything.
Now mirrors `DSRLAgent`: gates RL and HL independently and delegates HL to
`steervla_actor.update_hl`.

### 2.2 Best-of-N selection was gated on RL updates

`impls/main_carla.py:2889`. The rollout dispatch read:

```python
if rl_updates_on and hasattr(agent, "sample_actions_with_vla"):
```

That rationale is DSRL-specific — DSRL's `sample_actions_with_vla` uses a learned noise actor
that is uncalibrated until RL trains it. But for `BestOfNAgent` that method **is** the policy,
and it scores candidates with a *pretrained* critic, so it must run regardless. With
`enable_updates_rl=False`, execution fell through to the plain fixed-noise VLA branch and
**no candidate selection ever happened** — silently, with no warning. Fixed:

```python
_is_best_of_n = str(agent_config.get("agent_name", "")) == "best_of_n"
...
if (rl_updates_on or _is_best_of_n) and hasattr(agent, "sample_actions_with_vla"):
```

**Detection:** zero `[best_of_n] selected candidate` lines across a 4,340-step run. That line
prints on every selection, so its absence is the signal to check.

---

## 3. Cross-route summary (completed runs)

Two routes finished 20,000 steps each with both fixes applied. The labeling distribution is
**remarkably stable across two very different scenarios**:

| Route | Eps | Chunks | BAD | GOOD | unlabeled | precursor (of BAD) | events B/G |
|---|---:|---:|---:|---:|---:|---:|---:|
| `enter-actor-flow-004` | 114 | 2,044 | 67.4% | 12.1% | 420 | 47.7% | 399 / 176 |
| `signalized-junction-left-turn-001` | 129 | 2,054 | 66.6% | 12.0% | 441 | 48.8% | 391 / 144 |

Both runs: 20,000 Best-of-N selections (1 per step), 395 HL gradient updates.

Two readings, and they point in opposite directions:

* **Encouraging** — CAST's causal credit assignment contributes ~48% of corrective supervision
  on both routes, consistently. That is supervision no temporal-overlap scheme would produce.
* **Concerning** — BAD/GOOD/unlabeled shares match to within ~1 point across two scenarios with
  different geometry, traffic, and failure modes. Genuine per-route driving quality would be
  unlikely to land that close. This is more consistent with the VLM applying a roughly fixed
  prior ("~2/3 of chunks are BAD") than with it measuring route-specific behavior. **Validate
  the labels against debug video before treating BAD as ground truth** — if the rate is
  prompt-induced, the HL update is training on largely undifferentiated corrective signal.

The 395 HL updates being identical across both runs is expected, not suspicious: the cadence is
deterministic (`update_interval=10` × `hl_update_every=5` over 20,000 steps).

---

## 3a. COMPLETED RUN — `enter-actor-flow-004`, 20,000 steps

First full run with **both fixes applied** (Best-of-N selection live, HL update wired).
Run dir: `..._enter-actor-flow-004_seed_0_20260804_005038`.

| | |
|---|---:|
| Env steps | 20,000 / 20,000 |
| Best-of-N selections | 20,000 (1 per step) |
| HL gradient updates | 395 |
| Episodes | 114 |
| CAST windows | 206 |
| Labeled action chunks | 2,044 |
| HL samples (`.npz`) | 2,044 |
| Episode videos | 230 |

### Chunk labels

| Label | Chunks | Share |
|---|---:|---:|
| BAD | 1,377 | 67.4% |
| unlabeled | 420 | 20.5% |
| GOOD | 247 | 12.1% |

### Credit assignment on BAD chunks

| Source | Chunks | Share of BAD |
|---|---:|---:|
| `direct` | 720 | 52.3% |
| `precursor` | 657 | 47.7% |

VLM events: 399 BAD / 176 GOOD.

**The headline result is the precursor share: 47.7%.** Nearly half of all corrective supervision
lands on chunks with *no overlapping event* — reachable only through CAST's causal credit
assignment, not through temporal alignment. In the earlier (pre-fix, plain-VLA-rollout) sample
this was 31.2%, so under Best-of-N selection the VLM attributes substantially more blame to
lead-up behavior. Suggestive, not controlled — different rollout policy *and* far more data.

The 67.4% BAD rate is close to the earlier 71.5%, i.e. Best-of-N selection did **not**
noticeably reduce how often the VLM judges behavior as bad. Worth understanding before treating
BAD labels as ground truth (see open questions).

### Run termination

The run reached step 20,000 and then died with exit 134 during route teardown:

```
> Stopping the route (wrapper)
terminate called after throwing an instance of 'std::runtime_error'
  what(): Responding error from function set_actor_simulate_physics:
          Actor could not be found in the registry. Actor Id: 18328
```

This is a **shutdown race, not a training failure** — all 20,000 steps and every artifact were
written first. But `run_carla.sh` sees exit ≥128 and relaunches the whole 20k run
(`attempt 1/50`). Stopped manually. **A completed run that segfaults on teardown will silently
restart from zero** unless someone intervenes; worth a guard on
`step >= online_steps` before the retry loop fires.

---

## 3b. CAST relabel results (earlier partial run)

Route `enter-actor-flow-004`, 4,340 env steps, 2 episodes.
**Caveat:** rollout policy was plain VLA, not Best-of-N (see §2.2).

### Chunk labeling — 29 windows, 435 chunks

| Label | Chunks | Share |
|---|---:|---:|
| BAD | 311 | 71.5% |
| GOOD | 68 | 15.6% |
| unlabeled (null) | 56 | 12.9% |

### Credit assignment on BAD chunks

| Source | Chunks | Share of BAD |
|---|---:|---:|
| `direct` (event overlaps chunk) | 214 | 68.8% |
| `precursor` (lead-up to event) | 97 | 31.2% |

The causal credit path is doing real work — nearly a third of corrective supervision lands on
chunks with **no overlapping event**, which is the whole point of CAST's credit assignment
over naive temporal alignment.

### VLM events

| Label | Events |
|---|---:|
| BAD | 53 |
| GOOD | 25 |

### Artifacts written

| Artifact | Count |
|---|---:|
| CAST windows (`cast_relabel.json` + `rollout.mp4` + `trajectory.json`) | 29 |
| HL samples (`.npz`) | 435 |
| `hl_samples.json` manifests | 29 |
| HL update batch dumps (JSON + PNG panel + decoded tokens) | 245 |
| Episode videos | 4 |

Label kinds in the HL dataset match the chunk labels exactly (68 GOOD / 311 BAD / 56 n/a),
confirming `store_good_chunks=True` is persisting reinforcing samples alongside corrective ones.

### Example window (`ep0001_win0001`)

VLM returned 4 GOOD events; per-chunk relabeling still rewrote subtasks toward more specific
behavior. Chunk 0:

- **original:** `The vehicle accelerates normally to reach the speed limit while maintaining a steady course.`
- **suggested:** `The vehicle accelerates steadily to follow the front car, maintaining a straight course.`

Mean suggestions per chunk: 0.99 (config asks for up to 3); 20 of 435 chunks got none.

---

## 4. High-level (VLM backbone) training

22 gradient steps over ~1,400 env steps, cadence as configured
(`hl_update_every=5`, `hl_update_num_steps=2`).

### Batch composition (first batch, 16 rows)

```
pool -> count: {online: 11, simlingo: 3, simplified_reasoning: 2}
online label split: BAD/precursor -> 9, GOOD/null -> 2
corrective split:   precursor -> 0, direct BAD -> 9
```

Online share ≈ 0.69 against the configured `hl_online_weight=0.7`, and the BAD-vs-GOOD draw
matches `hl_online_bad_fraction=0.8`. The replay mixture is loading from
`/raid/users/cglossop/steervla_hl_pools`, so backbone drift protection is active.

A later batch reached the configured 0.5 precursor/direct balance; the first batch's
`precursor -> 0` was a cold-pool artifact, topped up from direct BAD as designed.

### Memory — the binding constraint

`hl_update_batch_size` is the knob that matters. Measured on an 80 GB H100 with
`hl_freeze_regexes=[".*img.*", ".*embedder.*"]`:

| Batch | Compile-time peak | Runtime peak | Result |
|---:|---:|---:|---|
| 64 | 71.65 GiB | — | **OOM → SIGABRT** |
| 16 | 52.51 GiB | 59.62 GiB | OK (limit 63.77 GB) |

Fixed floor is **18.9 GiB** (trainable params + Adam mu/nu) which rematerialization cannot
reduce. Runtime peak was **identical across all 22 updates** (`peak=59.62GB`), so padded shapes
are static and the peak is deterministic — but headroom is only **4.15 GB (93.5% utilization)**.
Batch 32 would not have fit.

The sibling `steervla_cast_relabel_config.py` comment advising "start at 128 and bump toward
256" does **not** hold for this checkpoint. Corrected figures are documented inline in the new
config.

> Raising `XLA_PYTHON_CLIENT_MEM_FRACTION` is not a safe workaround: it is process-global, so
> it would also expand JAX's claim on the sim GPU and starve the CARLA renderer sharing it.

---

## 5. Operational notes

Issues hit while launching, each with the fix:

| Symptom | Cause | Fix |
|---|---|---|
| `PermissionError: '/home/celinet'` at startup | `main_carla.py:147` hardcodes `--save_dir` to another user's home | pass `--save_dir=/raid/users/cglossop/carla_exps` |
| `TypeError` on first update | `BestOfNAgent.update_with_vla` lacked `global_step` | §2.1 |
| No candidate selection | RL-gated dispatch | §2.2 |
| OOM at first HL step | `hl_update_batch_size=64` | → 16 |
| `update_dagger` missing | `run_carla.sh` defaults `--train-mode dagger` | pass `--train-mode rl` |

**`--train-mode rl` is mandatory** for this config — the default routes updates to
`agent.update_dagger`, which `BestOfNAgent` does not implement.

Throughput ≈ 1.1 s/step raw, but ~40 steps/min sustained (CAST VLM queries block the rollout)
→ **~8 h for 20k steps**.

Other tenants were on GPUs 3/4 throughout; `reset_carla.sh` was deliberately **not** used
(it SIGKILLs every CARLA process on the box). `carla_job.sh stop <n>` is the scoped
alternative.

---

## Status

Three jobs relaunched with both fixes, 20,000 steps each, `--run-group BoN-CAST`,
`--train-mode rl --critic-mode none`, CoT temperature 0.5, `best_of_n=10`.

Final outcome of the three jobs:

| Job | Route | Render | Train | HL | Final state |
|---|---|---:|---:|---:|---|
| 5 | `enter-actor-flow-004` | 6 | 6 | 7 | **COMPLETE** — 20,000/20,000 (exit 134 on teardown only) |
| 6 | `signalized-junction-left-turn-001` | 3 | 3 | 4 | **COMPLETE** — 20,000/20,000, clean exit 0 |
| 7 | `opposite-vehicle-running-red-light-004` | 2 | 0 | 5 | stopped at 16,377/20,000 — never completed |

Job 7 reached 16,377 steps on the split-renderer topology (vs. hanging at 13,942 on the shared
one) but did not finish, so the split-renderer fix is **suggestive, not proven**.

### Hangs: CARLA renderer and JAX must not share a GPU

Two jobs (6 and 7) froze mid-run in `futex_wait_queue` with GPUs at 0% utilization, always
inside Best-of-N candidate sampling. VRAM at the time:

| GPUs | Role | Used |
|---|---|---:|
| 2, 3, 6 | CARLA renderer **+** JAX train | 97.4–98.8% |
| 4, 5, 7 | HL update only | 76.7% |

JAX preallocates ~75% (61 GB); CARLA's UE4 renderer grows on top of it across the run. When they
collide the allocation **blocks instead of erroring**, deadlocking on the most VRAM-hungry
per-step op (BoN action decoding). It builds slowly, which is why it strikes hours in.

**Fix: give the renderer its own GPU.** GPUs 0/1 cannot run Vulkan but are fine for JAX compute,
so `--train-gpu 0 --render-adapter 2 --hl-gpu 5` uses otherwise-dead capacity at no cost.
Applied to job 7.

### No checkpoints are written

`save_interval` defaults to **100,000** env steps (`main_carla.py:323`) — larger than a 20,000
step run, so `save_agent` never fires and `--resume` can recover nothing. Job 7's 13,942 steps
of agent state, including HL-fine-tuned backbone weights, were unrecoverable after its hang.
CAST windows and the HL dataset survive (written incrementally). **Set `save_interval < online_steps`
if the trained weights matter.**

### GPUs 0 and 1 cannot host the UE4 renderer on this box

Job 6 failed to start four times before landing on adapter 3. Measured behavior:

| Adapter | Result | Evidence |
|---:|---|---|
| 0 | `VulkanRHI::vkDeviceWaitIdle failed, VkResult=-4 VK_ERROR_DEVICE_LOST` → SIGSEGV | job 6 ×2 |
| 1 | same `VK_ERROR_DEVICE_LOST`; once a `GameThread timed out waiting for RenderThread after 60s` under memory pressure | job 6 ×2 |
| 2 | works | job 7 |
| 3 | works | job 6, and another tenant's runs |
| 6 | works | job 5 |

All failures surface as `RuntimeError: CARLA server failed to come up on rpc port 12600`
(`carla_utils.py:716`). The decisive test: adapter 1 failed with **1938 GB RAM free**, so this
is a GPU/Vulkan fault, not memory. Neither GPU reports `ERR!` or ECC errors and both work fine
for JAX compute — it is specifically UE4/Vulkan device creation that fails. Consistent with the
`reset_carla.sh` caveat about wedged GPUs needing `sudo nvidia-smi --gpu-reset -i <index>`
(untried; needs sudo).

**Use adapters 2, 3, or 6 for rendering.** GPUs 0/1 remain fine as HL/compute targets.

Note: JAX creates a ~528 MiB context on *every* visible GPU per job, because `run_carla.sh`
forces `use_cuda_visible_devices: False`. Harmless, but it means "idle" GPUs never read as 0 MiB.

### Separately: host RAM exhaustion killed job 5 mid-run

At 23:57 another user's single `CarlaUE4-Linux-Shipping` process reached **1888.5 GB RSS**
(94% of the box's 2 TB; all three of our jobs together used 37 GB). Job 5's CARLA server
segfaulted, leaving `CarlaUE4.sh` a zombie while `main_carla.py` stayed alive at 16% CPU,
blocked forever on an RPC to the dead server.

**The crash supervisor cannot catch this.** `run_carla.sh` retries on exit code ≥128; a process
that *hangs* never exits, so it is invisible — and equally invisible to a log monitor watching
for crash strings, since none are printed. Job 5 sat wedged for 52 minutes, losing everything
past step 114, while job 7 ran normally on the same box.

Mitigation now in place: a stall detector checking log mtime for every job every 2 min, alerting
on 15+ min of silence regardless of process state. **Recommend adding an equivalent watchdog to
`run_carla.sh`** — "no log output for N minutes" is the only signal that catches this class.

W&B project `catglossop/OGBench-CARLA`. Artifacts under
`/raid/users/cglossop/carla_exps/OGBench-CARLA/BoN-CAST/`.

**Verify Best-of-N is live before trusting any new results:**

```bash
grep -c "best_of_n\] selected candidate" .run_carla/jobs/job-5.log   # must be > 0
```

### Fix confirmed

Job 5 post-fix: **27 selections in 27 env steps** (one per step, as configured by
`actions_per_model_query=1` / `actions_per_cot=1`).

| Metric | Value |
|---|---|
| Winner index histogram (0–9) | `{1:1, 2:3, 3:1, 4:3, 5:2, 6:6, 7:4, 8:2, 9:5}` |
| Winning Q | mean −6.02, range [−77.06, 16.03] |
| Per-step candidate Q | median-of-min −66.0 · median-of-median −34.2 · median-of-max −2.1 |
| Action-chunk entropy (mean) | −2.663, range [−3.08, −0.24] |

Winners land across all 10 slots with no collapse onto index 0, so CoT sampling at
temperature 0.5 produces genuinely diverse candidates and the critic discriminates among them.

**Caveat on the Q spread.** Most candidates score −60 to −120 while the winner sits near −2.
That gap is wide enough that the critic may be rejecting out-of-distribution chunks rather
than making fine-grained quality judgments among plausible ones. Selection would still be
sensible, but it would mean Best-of-N is acting as an OOD filter, not a quality ranker.
Untested — compare selected vs. rejected chunks in the `bon_viz_interval=100` debug panels
before drawing conclusions.

### Open questions

1. **BAD rate is 71.5%.** Either the policy genuinely drives poorly on this route, or the VLM
   prompt is biased toward finding fault. Worth checking a debug video against its labels
   before treating BAD chunks as ground truth.
2. **Unconditioned ranking.** With the pretrained critic forcing `lang_dim=0`, candidate
   subtasks do not influence selection. Whether the subtask-conditioned variant selects
   differently is untested.
3. **No RL.** `enable_updates_rl=False` throughout, so the critic never adapts to the
   relabeled data — the only learning signal is the HL backbone update.
