# CAST offline collection → RLDS — implementation report

**Status: implemented and tested end-to-end on synthetic data. Not yet run against CARLA.**
Date: 2026-08-05.

The variant asked for: instead of interleaving VLM relabeling with `update_hl` gradient steps
during the rollout, **roll out ~20k steps with a frozen policy, save every relabeled chunk to
disk, then fine-tune offline in steervla-pi at a large batch size**. This report covers the two
halves that are done — the data-saving side in CARLA, and the RLDS converter — and lists the
open questions for the training discussion.

## Contents

| Path | What it is |
|---|---|
| `impls/configs/steervla_cast_collect_config.py` | **new** — collection-only agent config |
| `impls/vlas/cast_hl_to_rlds.py` | **new** — collected corpus → RLDS/TFDS converter |
| `impls/coaches/cast_relabel.py` | modified — richer HL sample schema + shared corpus root |
| `impls/main_carla.py` | modified — two extra fields passed to `record_model_input`, run tag |

---

## 1. Collection side

### What already existed

`OnlineCastRelabelSession` already wrote `steervla_hl_dataset_format` samples to disk — that is
how the *online* `SteerVLAActor.update_hl` is fed. Per relabel window it writes

```
<hl_dataset_dir>/<window tag>/sample_0000.npz    # image, state, current_speed, actions, action_loss_mask
<hl_dataset_dir>/<window tag>/hl_samples.json    # per-sample subtask/reasoning targets + provenance
```

So the *mechanism* was there. What it did not have was (a) enough per-sample context to
reconstruct an RLDS frame, and (b) any way to accumulate across runs. Both were added rather than
duplicated, so the offline corpus is byte-compatible with the online reader: a collection
directory can still be handed to `update_hl` unchanged.

### Config: `steervla_cast_collect_config.py`

Inherits `steervla_cast_relabel_config.py` and changes only what collection requires:

- `enable_updates = False` plus all three per-kind switches off. **No gradients at any point.**
  This matters beyond saving compute: a collection run must keep one fixed behavior policy from
  the first step to the last, or the corpus silently mixes samples from different policies.
- `steervla.load_trainable_params = False`, `hl_training_gpu_rank = -1`, replay pools cleared.
  The OpenPI `TrainState` (params + Adam mu/nu + backward activations) is the VRAM hog; collection
  only needs forward-only inference params, which frees the second GPU outright.
- `cast_relabel.store_hl_dataset = True`, `store_good_chunks = True` — save **both** halves. The
  BAD/GOOD balance is a training-time mixture decision, not something to bake into what we
  bothered to save.
- `cast_relabel.hl_dataset_root = "/raid/users/cglossop/cast_collect"` — absolute, shared across
  runs (see below).
- `cast_relabel.ego_history_len = 4`.

Launch (one per route; `run_carla.sh` already forwards `--agent-config` and `--online-steps`):

```bash
./run_carla.sh --agent-config impls/configs/steervla_cast_collect_config.py \
  --route parking-cut-in-001 --online-steps 20000 \
  --train-gpu 0 --render-adapter 1 --x-display-num 30
```

`GEMINI_API_KEY` must be exported — the relabel pass *is* the run; without it every window
review fails non-fatally and the corpus stays empty.

### Corpus layout: one root, many runs

`cast_relabel.hl_dataset_root` (new) overrides the default per-run
`<save_dir>/cast_relabel_hl_dataset`. When set, the session writes to `<root>/<run_tag>/`, where
`run_tag` is the run's `exp_name` (passed from `main_carla`):

```
/raid/users/cglossop/cast_collect/
  parking-cut-in-001-pi_prefix-sd000_.../
    ep0001_win0001/{sample_0000.npz, …, hl_samples.json}
    ep0001_win0002/…
    windows.jsonl                     # running per-window index
  merge-cut-in-002-.../
```

One level of nesting, deliberately: it prevents window-tag collisions across runs **and**
preserves the `<dir>/<window>/hl_samples.json` shape that `SteerVLAActor._scan_pool` globs, so
each run's subdirectory is still directly loadable by the online updater.

`windows.jsonl` is a new best-effort append-only index (route / episode / window / sample count /
BAD / precursor counts) so a collection run in progress can be inspected without walking every
manifest.

### Sample schema, extended (`hl_samples.json` `schema_version: 2`)

Everything the RLDS frame needs that was previously unrecoverable:

| Field | Where | Why it was needed |
|---|---|---|
| `ego_hist` | npz, `(4, 2)` f32 | RLDS `observation/ego_hist`. Raw `[speed m/s, yaw deg]` over the 4 env steps ending at the sample (indices 15 and 5 of the CARLA state vector — the same two `carla_state_vec_to_steervla_state` reads). Left-padded with the oldest pair in the first steps of an episode. |
| `routing_command` | manifest | The bare instruction. The stored `prompt` carries the `"The current speed is X m/s. "` prefix, and the RLDS loader **re-adds** that prefix from the `speed` field — storing the prefixed form would double it. Regex-stripping `prompt` is the fallback for old corpora. |
| `original_subtask` / `original_reasoning` | manifest | On the corrective path `subtask`/`reasoning` hold the VLM's replacement; keeping the model's own CoT lets a converted dataset report and filter on what actually changed. |
| `route`, `global_step`, `current_speed` | manifest | Provenance / grouping. |

The ego-history ring is pushed on **every** env step (not just chunk starts), so the retained
chunk-start sample carries a real 4-step history rather than a tiled current pair. Storing 4
pairs means one corpus can feed either the ego-history or the no-ego-history OpenPI config
without recollecting — the loader takes `[-2:]` when `include_ego_history=False`.

`main_carla.py` passes the two new values through to `record_model_input`
(`routing_command` from the obs dict, `global_step`), and hands the session `run_tag=exp_name`.

**Backwards compatibility:** the online reader (`_scan_pool` / `_record_from_entry`) resolves
every field with `.get()`, so extra npz keys and manifest fields are ignored. Old corpora
(schema 1) still load, and the converter falls back to tiling the current state pair and to
stripping the speed prefix.

---

## 2. RLDS converter — `impls/vlas/cast_hl_to_rlds.py`

Emits a TFDS dataset in the **SimLingo** RLDS layout that
`openpi.training.steervla_rlds_dataset._build_simlingo_restructure` reads, matching
`pi05_steervla_cot_simplified_reasoning_no_ego_history`.

### Schema

```
steps/observation/image                       jpeg, (224, 224, 3)
steps/observation/ego_hist                    (4, 2) f32   raw [speed m/s, yaw deg]
steps/speed                                   f32          current speed, m/s
steps/routing_command                         text         bare instruction
steps/action/future_10_xy_delta_t             (10, 2) f32  meters; loader divides by 7.0
steps/action/future_10_xy_delta_space         (10, 2) f32  loader uses as-is
steps/action/future_10_speed_course_delta_t   (10, 2) f32  ZERO PLACEHOLDER
steps/action/future_10_course_delta_space     (10,)   f32  ZERO PLACEHOLDER
steps/prompt                                  text         CoT *subtask* target
steps/gemini_refined_label                    text         CoT *reasoning* target
steps/cast_{label,credit_source,action_matches_subtask,original_subtask,episode_step,chunk_index}
episode_metadata/{run_tag, route, episode, window_index, num_steps}
```

Three schema decisions worth flagging:

**CoT field names are read from the OpenPI config, not hardcoded.** `--actor-config` resolves
`hl_cot_subtask_key` / `hl_cot_reasoning_key` off the `TrainConfig`, the same trick
`extract_hl_replay.py` uses. For the target config those are `prompt` (subtask) and
`gemini_refined_label` (reasoning) — confirmed by resolving the real config. Note the naming trap:
the RLDS field called `prompt` is the **subtask**, not the model's input prompt. The model's
prompt is rebuilt by the loader from `routing_command` + `speed`, and the reconstruction is
character-identical to what `routing_instruction_prompt` produced online
(`"The current speed is 3.4 m/s. Turn right in 20 meter."`, verified).

**The action split is an identity.** The chunk we store is the executed OpenPI chunk *already
denormalized to physical units* by `_postprocess_action_trajectory`
(`denormalize_actions`: `[:, :2] * 7`, `[:, 2:]` untouched for `DELTA_XY_T_DELTA_XY_SPACE`) —
which is exactly the raw RLDS action. So columns 0:2 → `future_10_xy_delta_t`, columns 2:4 →
`future_10_xy_delta_space`, and pushing the dataset back through the loader reproduces the
model-space action bit for bit. Verified numerically.

**`future_10_speed_course_delta_t` / `future_10_course_delta_space` are zeros.** The rollout only
ever produces the xy formulation, and under `DELTA_XY_T_DELTA_XY_SPACE` the loader computes then
discards those two. Training with any *other* `output_action_format` on this dataset would
silently learn zeros, so the converter **errors out** if the resolved config uses a different
format, unless `--allow-placeholder-actions` is passed.

**Images at 224×224, stretched (not padded).** 224 is what the model consumes
(`ResizeImages(224, 224)`) and what the online HL path feeds it, so storing at 224 makes the
offline pipeline pixel-identical to the online one instead of round-tripping through 512. The
stretch (not a letterbox pad) matches the pretraining preprocessing — a pad would feed the
backbone black bars it never saw. Source frames are 144×256.

### Supervision split

Whether the action is *supervised* is a training-time choice — the loader masks it
(`action_supervision=False`) for anything in `hl_dataset_name_weight_mappings` and supervises
anything in `dataset_name_weight_mappings`. `--split supervision` therefore emits **two**
datasets so that choice can be made per half:

- `<name>_corrective` — subtask replaced by the VLM; the executed action no longer matches it, so
  it must stay action-masked. Register under `hl_dataset_name_weight_mappings`.
- `<name>_reinforce` — original subtask kept; action and subtask are consistent, so it *may* be
  action-supervised. Register under either mapping.

Default is `--split none` (one combined HL dataset), matching what the online path does today.

### Usage

```bash
# in a TF-equipped env (the CARLA venv deliberately has no TF, same as extract_hl_replay.py)
uv run --with tensorflow-cpu --with tensorflow_datasets \
  python impls/vlas/cast_hl_to_rlds.py \
    --hl-root /raid/users/cglossop/cast_collect \
    --out-dir /raid/datasets/steervla \
    --dataset-name cast_relabel_hl_v1 \
    --actor-config pi05_steervla_cot_simplified_reasoning_no_ego_history
```

`--dry-run` needs neither TF nor openpi and reports counts / GOOD-BAD-precursor split / per-route
breakdown — useful to check a collection run mid-flight. Other flags: `--hl-root` repeats to merge
corpora, `--episode-key {episode,window}`, `--keep {all,corrective,reinforcing}`, `--limit`,
`--min-subtask-chars`, `--image-size`, `--ego-history-len`, `--version`.

Then register it:

```python
hl_dataset_name_weight_mappings={
    "simplified_reasoning_dataset": 1.85,
    "cast_relabel_hl_v1": 1.0,          # ← weight is the open question, see below
},
rlds_data_dir="/raid/datasets/steervla",
```

No builder class needs to be importable at training time. `tfds.builder(name, data_dir=...)` —
what the OpenPI loader calls — falls back to `builder_from_files`, reconstructing the dataset
from the `features.json` written on disk. Confirmed: reading back gives a `ReadOnlyBuilder`.

A `cast_conversion_stats.json` is written next to the built dataset (sample counts, label split,
per-route/per-run breakdown, the placeholder-field list, and the conversion parameters).

---

## 3. What was tested

The TFDS half was exercised in a scratch venv (`tensorflow-cpu 2.21` + `tfds 4.9.10`) against a
synthetic corpus written through the **real** `write_hl_samples` path:

- `OnlineCastRelabelSession` with `hl_dataset_root` set → writes to `<root>/<run_tag>/`; ego
  history is the true `t-3…t` window, left-padded correctly at episode start; chunk-start capture
  every 10 steps; `windows.jsonl` appended.
- `build_hl_samples_from_window` → corrective (BAD/precursor → replacement subtask) and
  reinforcing (GOOD → original subtask) targets, with route/global_step/ego_hist populated.
- Converter discovery, filtering (`--keep`), episode grouping, `--dry-run` in the CARLA venv.
- Full TFDS build, both `--split none` and `--split supervision`.
- **Read-back through the exact ops the OpenPI SIMLINGO restructure performs**: `tfds.builder`
  read-only fallback → `SkipDecoding` on `steps` → `tf.io.decode_image` gives `(224,224,3) uint8`
  → `ego_hist` → `(T,2)` ego state → action concat → `(T,10,4)` → prompt reconstruction →
  `"The current speed is 3.4 m/s. Turn right in 20 meter."` → subtask/reasoning off
  `prompt`/`gemini_refined_label`.
- `resolve_openpi_params("pi05_steervla_cot_simplified_reasoning_no_ego_history")` against the
  installed openpi → `subtask_field=prompt`, `reasoning_field=gemini_refined_label`,
  `output_action_format=DELTA_XY_T_DELTA_XY_SPACE`, `action_chunk_size=10`.
- `ruff check` clean on the new files; the two modified files gained no new findings.

**Not tested:** an actual CARLA collection run (needs a live sim + `GEMINI_API_KEY`), and training
on the produced dataset in steervla-pi. dlimp was not installed in the scratch env, so the
restructure ops were replayed by hand rather than through `dl.DLataset.from_rlds`; the fields and
decode path they touch are all covered above.

**Unrelated pre-existing bug found in passing:** `cast_relabel.build_debug_task_prompt` references
`{seed_block}` but only defines `seed_subtask_block` / `seed_reasoning_block` — `debug_task=True`
would raise `NameError`. Left alone (the collect config sets `debug_task=False`); flagging it
because it means the debug-task path is currently dead.

---

## 4. Open questions for the training discussion

1. **Mixture weight** for `cast_relabel_hl_v1` against `simplified_reasoning_dataset` (1.85) and
   the 16 SimLingo datasets. The online path used online 0.7 / simlingo 0.2 / simplified 0.1 with
   an 80/20 BAD/GOOD bias *inside* the online share and a further 50/50 direct/precursor split.
   None of that intra-pool bucketing exists in the RLDS loader — it samples uniformly within a
   dataset. Reproducing it means either splitting into more datasets (BAD-direct /
   BAD-precursor / GOOD as three named datasets, weighted) or accepting a uniform draw.
   `--split supervision` is the two-way version of this; a four-way split is a small change.
2. **Action supervision on the reinforce half** — free extra signal, or a way to lock in the
   current policy's mistakes on chunks the VLM merely failed to flag?
3. **How much data.** 20k steps at `action_horizon=10` is ≤2k chunks per run *before* the VLM
   drops anything; windows that fail review contribute nothing. Number of routes × runs needed to
   fill a large-batch fine-tune is worth estimating from the first collection run's yield
   (`windows.jsonl` + `--dry-run` give this directly).
4. **Whether to keep `store_good_chunks`** in the corpus but filter at conversion time (currently
   what `--keep` does) or filter at collection.
