"""Model-agnostic high-level (HL) replay-pool sampling shared by the online HL-update actors.

Moved verbatim out of :class:`vlas.steervla.SteerVLAActor` so a second actor (the torch SimLingo
SteerVLA in :mod:`vlas.simlingo_steervla`) draws its HL batches with exactly the same logic: the
online cast_relabel pool + offline replay pools scan, the BAD/GOOD (and precursor) bucketing, the
adaptive severity weights, the ``hl_min_online_samples`` gate, and the sample-reuse telemetry.

Nothing here touches a model. The host class supplies ``_read_hl_record`` (what one sample's arrays
and supervision look like for that model) and the attributes the methods read (``hl_dataset_dir``,
``hl_online_weight``, ``_hl_replay_pool_specs``, ...).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

# ── Adaptive online sampling (``steervla.use_adaptive_sampling``) ─────────────────────
# Per-sample draw weights for the ONLINE cast_relabel pool. Off by default; when
# ``use_adaptive_sampling`` is set, the online pool's entries are drawn with probability
# proportional to these instead of uniformly at random.
#
# Two things stay separate on purpose:
#
#   * **Coverage** between correcting bad behavior and reinforcing good behavior is still governed
#     by ``hl_online_bad_fraction`` — the corrective/reinforce split is a *quota*, not something the
#     weights are allowed to erode. Turning adaptive sampling on does not change how many corrective
#     rows a batch gets, only *which* ones.
#   * **Severity** within each of those two buckets is what these weights express.
#
# Because the draw is normalized inside each bucket, only ratios *within* a bucket are meaningful:
# ``good_success`` does not compete with ``bad_precursor_catastrophic``; it competes with ``good``.
#
# The ordering encodes the intent: a chunk that causally set up a catastrophe (a collision, or the
# ego abandoning the routing command) is the most informative correction available, an ordinary
# precursor next, and the directly-blamed chunk least — by the time a chunk is directly overlapping
# the failure the mistake is usually already unavoidable, so it teaches less than its lead-up does.
# On the reinforce side, a chunk from an episode that actually completed the route is worth more
# than a GOOD chunk from a mediocre one, and an unlabeled chunk carries no positive signal at all.
ADAPTIVE_SAMPLING_WEIGHTS: dict[str, float] = {
    # corrective bucket (label == "BAD")
    "bad_precursor_catastrophic": 4.0,
    "bad_precursor": 2.0,
    "bad_direct_catastrophic": 1.0,
    "bad_direct": 0.5,
    # reinforce bucket (label GOOD or absent)
    "good_success": 3.0,
    "good": 1.0,
    "unlabeled": 0.5,
}

# Outcome tags written per sample by ``coaches/cast_relabel.py::resolve_window_outcome``. Restated
# here rather than imported: ``steervla.py`` is also imported by the inference server, which has no
# reason to pull in the coach package. Kept in sync by the assertion in that module's docstring —
# an unknown tag is treated as "unremarkable", never as an error, so an older corpus still trains.
_ADAPTIVE_CATASTROPHIC_OUTCOMES = frozenset(
    {"collision", "crash_stuck", "off_route", "route_divergence"}
)
_ADAPTIVE_SUCCESS_OUTCOMES = frozenset({"success", "route_completed"})


class HLPoolSamplingMixin:
    """HL batch drawing over the online cast_relabel pool and the offline replay pools."""

    def _init_hl_pool_sampling(
        self,
        *,
        hl_replay_root: str | Path | None = None,
        hl_replay_pools: list[dict] | None = None,
        hl_online_weight: float = 1.0,
        hl_online_bad_fraction: float = -1.0,
        hl_online_precursor_fraction: float = -1.0,
        hl_online_backfill_from_replay: bool = True,
        use_adaptive_sampling: bool = False,
        adaptive_sampling_weights: dict | None = None,
        hl_min_online_samples: int = 1,
        hl_keep_last_rounds: int = 0,
    ) -> None:
        """Set the state the pool-sampling methods read. Hosts call this from ``__init__``."""
        # HL replay pools: a small amount of the original pretraining data (pre-extracted to npz by
        # ``impls/vlas/extract_hl_replay.py``) mixed into every HL update to stabilize the VLM
        # backbone. Weighted like steervla-pi's dataset mixture: the online cast_relabel pool gets
        # ``hl_online_weight`` and each replay pool its own weight; per HL update the batch is split
        # across pools by normalized weight. Each replay pool carries its own supervision flags
        # (``action_supervision`` = train the flow head; ``supervise_fast`` = train the FAST CE).
        self.hl_online_weight = float(hl_online_weight)
        # Within the online cast_relabel pool, bias the per-batch draw toward corrective chunks:
        # ``hl_online_bad_fraction`` of each online-share slot is filled from the BAD / BAD(precursor)
        # bucket (``label == "BAD"``, either ``credit_source``) and the remainder from the GOOD / null
        # bucket (``label`` GOOD or absent). Whichever bucket is short is topped up from the other so a
        # full batch is still produced. ``< 0`` disables the split (uniform draw over the online pool).
        # Only the online pool is bucketed; replay pools keep their weighted share untouched.
        self.hl_online_bad_fraction = float(hl_online_bad_fraction)
        # Within the corrective (BAD) bucket, further balance BAD(precursor) chunks
        # (``credit_source == "precursor"``) against direct BAD chunks: ``hl_online_precursor_fraction``
        # of the corrective slots are filled from the precursor sub-bucket and the remainder from the
        # direct sub-bucket, again topping up from whichever sub-bucket has spares. ``< 0`` disables the
        # sub-split (corrective slots drawn uniformly over BAD/precursor). Only meaningful when
        # ``hl_online_bad_fraction >= 0`` (the corrective bucket must exist to be sub-split).
        self.hl_online_precursor_fraction = float(hl_online_precursor_fraction)
        # What to do when the online pool cannot meet the ``hl_online_bad_fraction`` split -- e.g.
        # 32 slots at 0.9 wants 29 corrective + 3 reinforce, but only 5 corrective chunks exist yet.
        #
        #   True  (default): each online bucket contributes at most its own target. The online pool
        #                    hands back FEWER than its weighted share and the shortfall is filled
        #                    from the offline replay pools, so the corrective/reinforce ratio the
        #                    config asked for is preserved instead of being silently inverted.
        #   False (legacy):  the short bucket's slots are taken from the other online bucket, so the
        #                    online share is always filled but at whatever ratio happens to be
        #                    available.
        #
        # The legacy behavior is what made an early-run batch read 12 corrective / 18 reinforce
        # under a 0.9 corrective target: with the corrective bucket nearly empty, cross-fill floods
        # the batch with reinforce rows precisely when there is least to reinforce. Backfilling from
        # replay instead trades those rows for frozen pretraining data, which is the safer filler.
        self.hl_online_backfill_from_replay = bool(hl_online_backfill_from_replay)
        # Draw the online pool by per-sample severity weight instead of uniformly. The
        # corrective/reinforce coverage quota (``hl_online_bad_fraction``) is unchanged and still
        # applies; only the choice of rows *within* each bucket becomes weighted. See
        # ``ADAPTIVE_SAMPLING_WEIGHTS`` for the categories and the reasoning behind their order.
        #
        # This SUPERSEDES ``hl_online_precursor_fraction``: the precursor/direct balance is what the
        # weights already express, and applying a hard sub-split on top of them would silently cap
        # the very rows the weighting exists to promote. When both are set, the sub-split is ignored
        # and :meth:`_load_hl_batch` says so once at the first update.
        self.use_adaptive_sampling = bool(use_adaptive_sampling)
        self.adaptive_sampling_weights = dict(ADAPTIVE_SAMPLING_WEIGHTS)
        for key, value in (adaptive_sampling_weights or {}).items():
            if key not in ADAPTIVE_SAMPLING_WEIGHTS:
                raise ValueError(
                    f"adaptive_sampling_weights: unknown category {key!r}; expected one of "
                    f"{sorted(ADAPTIVE_SAMPLING_WEIGHTS)}"
                )
            if float(value) < 0.0:
                raise ValueError(f"adaptive_sampling_weights[{key!r}] must be >= 0, got {value!r}")
            self.adaptive_sampling_weights[key] = float(value)
        # One-shot log flags so the mode announces itself once per run rather than per update.
        self._adaptive_logged_once = False
        self._adaptive_no_outcomes_warned = False
        # How many online cast_relabel samples must exist before the FIRST HL update runs. The batch
        # does NOT wait for the online pool to fill its whole weighted share: as soon as this many
        # online samples are on disk, the update takes whatever the online pool has and fills the rest
        # of the batch from the offline replay pools (and, if those can't cover it either, by resampling
        # what's available). Set to 0 to allow replay-only updates before any online sample lands.
        self.hl_min_online_samples = max(0, int(hl_min_online_samples))
        # Pooled training: keep only samples from the last N policy versions (0 = keep everything).
        self.hl_keep_last_rounds = max(0, int(hl_keep_last_rounds))
        self._hl_replay_root: Path | None = Path(hl_replay_root) if hl_replay_root is not None else None
        self._hl_replay_pool_specs: list[dict[str, Any]] = self._resolve_replay_pool_specs(hl_replay_pools)
        self._hl_replay_logged_once = False
        self._hl_replay_missing_warned = False
        self._hl_update_calls = 0
        # --- Policy-update / sample-reuse accounting (overfitting telemetry) -------------------
        # ``_hl_update_calls`` above counts *attempts*, including the ones the ``hl_update_every``
        # throttle drops, so it is not "how much did the policy actually move". These three are:
        #   _hl_updates_applied -> update_hl bodies that reached the gradient loop.
        #   _hl_grad_steps      -> cumulative optimizer steps applied to the backbone (the real
        #                          "number of updates to the policy"; = sum of hl_update_num_steps).
        #   _hl_sample_uses     -> per-sample-id count of gradient steps that consumed it. A batch
        #                          row used by an ``ns``-step update counts ``ns`` times, and a row
        #                          duplicated by ``_pad_hl_batch`` counts once per copy, because
        #                          both really are extra gradient exposure for that sample.
        self._hl_updates_applied = 0
        self._hl_grad_steps = 0
        self._hl_sample_uses: dict[str, int] = {}
        self._hl_sample_uses_by_pool: dict[str, dict[str, int]] = {}
        # Diagnostics for :meth:`update_hl`, whose skip paths are otherwise silent (it returns
        # ``{}``), which makes an absent ``vla_hl/batch_text`` table impossible to explain.
        self._hl_pool_size = 0
        self._last_hl_skip_reason: str | None = None
        self._last_hl_note: str | None = None

    def _resolve_replay_pool_specs(self, hl_replay_pools) -> list[dict[str, Any]]:
        """Normalize the configured replay-pool list into ``{dir, name, weight, kind}`` specs.

        Each entry is a ``{name, weight}`` dict (``name`` is a dir under ``hl_replay_root`` or an
        absolute path). Supervision flags (``action_supervision`` / ``supervise_fast`` / state format)
        are NOT set here — they live in each pool's ``hl_samples.json`` (written by
        ``extract_hl_replay.py``) and are read at scan time.
        """
        specs: list[dict[str, Any]] = []
        for p in (hl_replay_pools or []):
            try:
                p = dict(p)
            except Exception:
                continue
            name = str(p.get("name") or p.get("dir") or "").strip()
            weight = float(p.get("weight", 0.0) or 0.0)
            if not name or weight <= 0.0:
                continue
            d = Path(name)
            if not d.is_absolute() and self._hl_replay_root is not None:
                d = self._hl_replay_root / name
            specs.append({"dir": d, "name": name, "weight": weight, "kind": "replay"})
        return specs

    def _hl_pools(self) -> list[dict[str, Any]]:
        """Active HL sources: the online cast_relabel pool plus any configured replay pools."""
        pools: list[dict[str, Any]] = []
        if self.hl_dataset_dir is not None and float(self.hl_online_weight) > 0.0:
            pools.append(
                {"dir": Path(self.hl_dataset_dir), "name": "online", "weight": float(self.hl_online_weight), "kind": "online"}
            )
        pools.extend(self._hl_replay_pool_specs)
        return [p for p in pools if float(p.get("weight", 0.0)) > 0.0]

    def _scan_pool(self, pool: dict[str, Any]) -> list[dict[str, Any]]:
        """List all samples in one pool, tagging each with the pool's supervision flags.

        The online cast_relabel dir keeps a ``hl_samples.json`` per window subdir; an extracted replay
        pool keeps a single ``hl_samples.json`` at its root — both globs are checked. A **pooled** run
        (``impls/cast_pool.py``) adds a third depth, ``<pool_root>/<worker>/<window>/``, so several
        rollout workers can write one shared corpus that the trainer reads whole. ``action_supervision``
        / ``supervise_fast`` / ``state_format`` come from the manifest (defaults match the online
        ``steervla_hl_dataset_format``: no flow, per-sample FAST, raw CARLA state).

        Window dirs still being written carry the ``cast_pool.TMP_PREFIX`` and are skipped, so a
        half-written manifest is never scanned.
        """
        entries: list[dict[str, Any]] = []
        root = pool.get("dir")
        if root is None or not Path(root).is_dir():
            return entries
        root = Path(root)
        manifests = (
            sorted(root.glob("hl_samples.json"))
            + sorted(root.glob("*/hl_samples.json"))
            + sorted(root.glob("*/*/hl_samples.json"))
        )
        manifests = [
            m for m in manifests if not any(p.name.startswith(".tmp-") for p in m.relative_to(root).parents)
        ]
        for manifest_path in manifests:
            try:
                manifest = json.loads(manifest_path.read_text())
            except Exception:
                continue
            work_dir = manifest_path.parent
            pool_action_sup = bool(manifest.get("action_supervision", False))
            pool_fast = manifest.get("supervise_fast", None)  # None -> resolve per-sample.
            state_format = str(manifest.get("state_format", "carla_raw"))
            for s in manifest.get("samples", []):
                entries.append(
                    {
                        "dir": work_dir,
                        "file": s.get("sample_file"),
                        "prompt": s.get("prompt", ""),
                        "subtask": s.get("subtask", ""),
                        "reasoning": s.get("reasoning", ""),
                        # The CoT the policy produced at rollout (cast_relabel keeps it even when
                        # ``reasoning`` holds the VLM replacement); "" for replay pools.
                        "original_reasoning": s.get("original_reasoning", ""),
                        # Only reinforced (original) online chunks have an action matching their subtask.
                        "action_matches_subtask": bool(s.get("action_matches_subtask", False)),
                        # cast_relabel per-chunk verdict: "GOOD"/"BAD"/None and (for BAD) whether the
                        # blame is "direct" or "precursor". Used to bucket the online pool 80/20
                        # (BAD or BAD-precursor vs GOOD or null) in :meth:`_load_hl_batch`. Replay
                        # pools have no such labels; they fall into the "not BAD" bucket harmlessly
                        # (the label split is only applied to the online pool).
                        "label": s.get("label"),
                        "credit_source": s.get("credit_source", ""),
                        # How the episode this sample came from ended, written by
                        # ``cast_relabel.resolve_window_outcome``. Only read when
                        # ``use_adaptive_sampling`` is on; "" (the default, and what every replay
                        # pool and every pre-2026-09-07 online corpus has) weights as unremarkable.
                        "outcome": str(s.get("outcome") or ""),
                        "action_supervision": pool_action_sup,
                        "supervise_fast": pool_fast,
                        "state_format": state_format,
                        "pool": pool.get("name", "online"),
                        # Which policy version produced this sample (pooled runs; see
                        # impls/cast_pool.py). -1 for pools that predate the field or for the
                        # offline replay pools, which are version-less and never age out.
                        # NOTE: version 0 (the pre-round base policy) is a real version — read it
                        # with an explicit None check, not ``or -1``, which would make 0 unversioned
                        # and thus permanently exempt from the staleness filter.
                        "policy_version": (
                            -1 if s.get("policy_version") is None else int(s["policy_version"])
                        ),
                    }
                )
        return entries

    def _filter_stale_entries(self, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop online samples produced more than ``hl_keep_last_rounds`` policy versions ago.

        Pooled training swaps the policy every round, so an old sample supervises a backbone that no
        longer produces the behavior it was correcting. Keeping a sliding window of versions bounds
        both that staleness and how many times any one sample can be re-drawn. Version ``-1``
        (unversioned pools, including every offline replay pool) is always kept -- those are
        pretraining data, not on-policy corrections.
        """
        keep = int(self.hl_keep_last_rounds)
        if keep <= 0:
            return entries
        versions = [int(e.get("policy_version", -1)) for e in entries]
        newest = max((v for v in versions if v >= 0), default=-1)
        if newest < 0:
            return entries  # nothing versioned; nothing to age out.
        floor = newest - keep + 1
        return [e for e in entries if int(e.get("policy_version", -1)) < 0 or int(e["policy_version"]) >= floor]

    @staticmethod
    def _largest_remainder(total: int, fracs: list[float]) -> list[int]:
        """Split ``total`` into integer per-source counts closest to ``fracs`` (sum == total)."""
        raw = [f * total for f in fracs]
        floors = [int(np.floor(x)) for x in raw]
        rem = int(total - sum(floors))
        if rem > 0:
            order = np.argsort([-(raw[i] - floors[i]) for i in range(len(raw))])
            for k in range(rem):
                floors[int(order[k % len(order)])] += 1
        return floors

    @staticmethod
    def _is_bad_entry(e: dict[str, Any]) -> bool:
        """True for a cast_relabel BAD / BAD(precursor) chunk (either ``credit_source``)."""
        return str(e.get("label") or "").strip().upper() == "BAD"

    @staticmethod
    def _is_precursor_entry(e: dict[str, Any]) -> bool:
        """True for a cast_relabel BAD(precursor) chunk (``credit_source == "precursor"``)."""
        return str(e.get("credit_source") or "").strip().lower() == "precursor"

    @classmethod
    def _adaptive_category(cls, e: dict[str, Any]) -> str:
        """Which :data:`ADAPTIVE_SAMPLING_WEIGHTS` bucket one online entry falls in.

        Resolution is total — every entry gets exactly one category — and unknown/absent outcome
        tags fall back to the non-catastrophic, non-success variant, so a corpus written before
        outcomes were recorded still draws (uniformly within its bucket, which is the old behavior).
        """
        outcome = str(e.get("outcome") or "").strip().lower()
        if cls._is_bad_entry(e):
            catastrophic = outcome in _ADAPTIVE_CATASTROPHIC_OUTCOMES
            if cls._is_precursor_entry(e):
                return "bad_precursor_catastrophic" if catastrophic else "bad_precursor"
            return "bad_direct_catastrophic" if catastrophic else "bad_direct"
        # Reinforce bucket. An unlabeled chunk is not evidence of good driving — the VLM simply had
        # nothing to say about it — so it is categorised apart from a chunk actually marked GOOD,
        # even when it comes from a successful episode.
        if str(e.get("label") or "").strip().upper() != "GOOD":
            return "unlabeled"
        return "good_success" if outcome in _ADAPTIVE_SUCCESS_OUTCOMES else "good"

    def _adaptive_weights_for(self, entries: list[dict[str, Any]]) -> np.ndarray:
        """Per-entry draw weight, as a float array aligned with ``entries``."""
        table = self.adaptive_sampling_weights
        return np.array(
            [float(table.get(self._adaptive_category(e), 0.0)) for e in entries], dtype=np.float64
        )

    @staticmethod
    def _weighted_order(entries: list[dict[str, Any]], weights: np.ndarray) -> list[dict[str, Any]]:
        """Random permutation of ``entries`` in which higher-weight entries tend to come first.

        Efraimidis-Spirakis: draw ``key_i = -log(u_i) / w_i`` and sort ascending. Taking the first
        k of that order is exactly weighted sampling *without replacement* with probabilities
        proportional to ``w``, for every k at once — which is what the caller needs, since it wants
        both a selection and a ranked list of leftovers to top up from.

        Handles the degenerate cases the obvious ``np.random.choice(p=...)`` does not: a zero weight
        sorts last instead of raising "fewer non-zero entries in p than size", and an all-zero
        weight vector degrades to a plain shuffle rather than a division error.
        """
        n = len(entries)
        if n == 0:
            return []
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (n,) or not np.any(w > 0.0):
            return [entries[int(j)] for j in np.random.permutation(n)]
        with np.errstate(divide="ignore"):
            # u in (0, 1] keeps -log(u) finite and >= 0; w == 0 -> key = inf -> sorted last.
            keys = -np.log(1.0 - np.random.random(n)) / w
        return [entries[int(j)] for j in np.argsort(keys, kind="stable")]

    def _order_online_entries_adaptive(
        self, entries: list[dict[str, Any]], cnt: int, bad_fraction: float
    ) -> tuple[list[dict[str, Any]], int]:
        """Adaptive-sampling counterpart of :meth:`_order_online_entries`.

        Same contract — ``(permutation, n_take)``, and the same corrective/reinforce coverage quota
        from :meth:`_online_bucket_counts`. The only difference is that each bucket is ordered by
        :meth:`_weighted_order` instead of shuffled uniformly, so severity decides *which* corrective
        rows fill the corrective quota rather than chance.

        ``hl_online_precursor_fraction`` is deliberately not applied here; see the constructor.
        """
        bad = [e for e in entries if self._is_bad_entry(e)]
        good = [e for e in entries if not self._is_bad_entry(e)]
        bad = self._weighted_order(bad, self._adaptive_weights_for(bad))
        good = self._weighted_order(good, self._adaptive_weights_for(good))
        # Identical quota arithmetic to the uniform path (including the backfill decision).
        n_bad, n_good = self._online_bucket_counts(bad, good, cnt, bad_fraction)
        selected = bad[:n_bad] + good[:n_good]
        leftovers = bad[n_bad:] + good[n_good:]
        # Shuffle only the selection's internal order (the batch is shuffled again downstream);
        # leftovers keep their weighted ranking so a top-up still takes the most informative spares.
        np.random.shuffle(selected)
        return selected + leftovers, len(selected)

    @staticmethod
    def _balance_buckets(
        primary: list[dict[str, Any]],
        secondary: list[dict[str, Any]],
        cnt: int,
        primary_fraction: float,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Pick up to ``cnt`` entries — ~``primary_fraction`` from ``primary``, the rest from
        ``secondary`` — topping up from whichever bucket has spares when the other underfills.
        Both inputs are assumed pre-shuffled. Returns ``(selected, leftovers)``.
        """
        n_primary = min(len(primary), int(round(cnt * float(primary_fraction))))
        n_secondary = cnt - n_primary
        if n_secondary > len(secondary):  # secondary bucket short -> pull more primary to reach cnt.
            n_secondary = len(secondary)
            n_primary = min(len(primary), cnt - n_secondary)
        selected = primary[:n_primary] + secondary[:n_secondary]
        leftovers = primary[n_primary:] + secondary[n_secondary:]
        return selected, leftovers

    def _online_bucket_counts(
        self,
        bad: list[dict[str, Any]],
        good: list[dict[str, Any]],
        cnt: int,
        bad_fraction: float,
    ) -> tuple[int, int]:
        """How many corrective / reinforce rows the online pool contributes to a ``cnt``-row share.

        Targets are ``round(cnt * bad_fraction)`` corrective and the remainder reinforce. Whether a
        bucket that cannot meet its target is topped up from the *other online bucket* or left short
        (for the caller to backfill from the offline replay pools) is
        ``hl_online_backfill_from_replay`` -- see the constructor.

        Returns ``(n_bad, n_good)``, which sums to ``cnt`` only when both buckets can cover their
        targets; under backfill it is deliberately allowed to sum to less.
        """
        n_bad_target = int(round(cnt * float(bad_fraction)))
        n_good_target = cnt - n_bad_target
        if self.hl_online_backfill_from_replay:
            return min(len(bad), n_bad_target), min(len(good), n_good_target)
        n_bad = min(len(bad), n_bad_target)
        n_good = cnt - n_bad
        if n_good > len(good):  # reinforce bucket short -> pull more corrective to reach cnt.
            n_good = len(good)
            n_bad = min(len(bad), cnt - n_good)
        return n_bad, n_good

    def _order_online_entries(
        self,
        entries: list[dict[str, Any]],
        cnt: int,
        bad_fraction: float,
        precursor_fraction: float = -1.0,
    ) -> tuple[list[dict[str, Any]], int]:
        """Order online-pool entries so the leading rows are ~``bad_fraction`` BAD, the rest GOOD/null.

        BAD and BAD(precursor) chunks form the corrective bucket; GOOD and unlabeled/null chunks the
        reinforce bucket; :meth:`_online_bucket_counts` sizes each. When ``precursor_fraction >= 0``,
        the corrective slots are themselves balanced between BAD(precursor) and direct BAD chunks
        (``round(n_bad * precursor_fraction)`` precursor, the rest direct), topping up from whichever
        sub-bucket has spares -- that cross-fill is kept even under backfill, because it moves rows
        *within* the corrective bucket and so cannot disturb the corrective/reinforce ratio.

        Returns ``(permutation, n_take)``: a full permutation of ``entries`` whose first ``n_take``
        rows are the selection, the rest ranked leftovers. ``n_take`` is ``cnt`` unless a bucket ran
        short under ``hl_online_backfill_from_replay``.
        """
        bad = [e for e in entries if self._is_bad_entry(e)]
        good = [e for e in entries if not self._is_bad_entry(e)]
        np.random.shuffle(bad)
        np.random.shuffle(good)
        n_bad, n_good = self._online_bucket_counts(bad, good, cnt, bad_fraction)
        if precursor_fraction >= 0.0:
            # Sub-split the corrective bucket by BAD(precursor) vs direct BAD. ``bad`` is already
            # shuffled, so the comprehensions inherit that shuffle.
            precursor = [e for e in bad if self._is_precursor_entry(e)]
            direct = [e for e in bad if not self._is_precursor_entry(e)]
            bad_selected, bad_leftover = self._balance_buckets(precursor, direct, n_bad, precursor_fraction)
        else:
            bad_selected, bad_leftover = bad[:n_bad], bad[n_bad:]
        selected = bad_selected + good[:n_good]
        leftovers = bad_leftover + good[n_good:]
        np.random.shuffle(selected)
        np.random.shuffle(leftovers)
        return selected + leftovers, len(selected)

    def _load_hl_batch(self, batch_size: int):
        """Draw a weighted mixed batch of HL records across the online + replay pools.

        Counts are split across active pools by their (normalized) weights, mirroring steervla-pi's
        dataset mixture. The online cast_relabel pool does NOT have to fill its whole weighted share:
        as soon as it holds ``hl_min_online_samples`` samples the update runs, taking every online
        sample available (up to its share) and topping the batch up from the offline replay pools.
        Returns ``None`` only when the online pool is still below ``hl_min_online_samples`` (or no
        pool has any readable sample at all).
        """
        bs = int(batch_size)
        pools = self._hl_pools()
        if not pools:
            self._hl_pool_size = 0
            return None
        scanned = [{"pool": p, "entries": self._filter_stale_entries(self._scan_pool(p))} for p in pools]
        # Warn once if replay pools were configured but none resolved on disk (extraction not run).
        if self._hl_replay_pool_specs and not getattr(self, "_hl_replay_missing_warned", False):
            replay_have = any(
                s["entries"] for s in scanned if s["pool"].get("kind") == "replay"
            )
            if not replay_have:
                self._hl_replay_missing_warned = True
                print(
                    "[steervla.update_hl] WARNING: hl_replay_pools configured but no replay samples "
                    f"found under {self._hl_replay_root} — training online-only. Run "
                    "impls/vlas/extract_hl_replay.py to populate the pools.",
                    flush=True,
                )
        online = next((s for s in scanned if s["pool"].get("kind") == "online"), None)
        # ``_hl_pool_size`` reports the *online* pool fill (what the skip messages talk about).
        self._hl_pool_size = (
            len(online["entries"]) if online is not None else sum(len(s["entries"]) for s in scanned)
        )
        # Gate only on *starting* the online stream, not on it filling its share: below
        # ``hl_min_online_samples`` we'd be training on replay alone, which is not the point of the
        # online HL update. At or above it, whatever is on disk goes in and replay covers the rest.
        if online is not None and len(online["entries"]) < self.hl_min_online_samples:
            return None

        active = [s for s in scanned if s["entries"]]
        if not active:
            return None
        wsum = sum(float(s["pool"]["weight"]) for s in active)
        counts = self._largest_remainder(bs, [float(s["pool"]["weight"]) / wsum for s in active])

        records: list[dict[str, Any]] = []
        # Spare candidates for topping the batch up when a pool underfills its share. Replay spares
        # are kept apart from online ones and used FIRST: when the online pool comes up short it is
        # because a bucket could not meet its target, so refilling from its own leftovers would put
        # back exactly the rows the quota just excluded. Online spares remain as a last resort ahead
        # of ``_pad_hl_batch``'s repeat-sampling, which is worse than any real row.
        leftovers: list[dict[str, Any]] = []
        online_leftovers: list[dict[str, Any]] = []
        online_short = 0
        for s, cnt in zip(active, counts):
            # The online pool is bucketed (corrective vs reinforce) when enabled; replay pools (and
            # a disabled split) draw uniformly at random.
            is_online = s["pool"].get("kind") == "online"
            if is_online and self.hl_online_bad_fraction >= 0.0:
                if self.use_adaptive_sampling:
                    self._log_adaptive_sampling_once(s["entries"])
                    ordered, take = self._order_online_entries_adaptive(
                        s["entries"], cnt, self.hl_online_bad_fraction
                    )
                else:
                    ordered, take = self._order_online_entries(
                        s["entries"], cnt, self.hl_online_bad_fraction, self.hl_online_precursor_fraction
                    )
                online_short = max(0, cnt - take)
            else:
                ordered = [s["entries"][int(j)] for j in np.random.permutation(len(s["entries"]))]
                take = cnt
            spares = online_leftovers if is_online else leftovers
            taken = 0
            for e in ordered:
                if taken >= take:
                    spares.append(e)
                    continue
                rec = self._read_hl_record(e)
                if rec is None:
                    continue
                records.append(rec)
                taken += 1
        for spares in (leftovers, online_leftovers):
            if len(records) >= bs or not spares:
                continue
            np.random.shuffle(spares)
            for e in spares:
                if len(records) >= bs:
                    break
                rec = self._read_hl_record(e)
                if rec is not None:
                    records.append(rec)
        if not records:
            return self._hl_short_batch(0, bs)
        records = records[:bs]
        # Records are assembled pool-by-pool (online block, then each replay pool), so without this
        # shuffle the batch — and the logged HL panel, which shows the leading rows — would be all
        # online. Interleave the pools so the batch (and its inspection panel) is a random mix.
        np.random.shuffle(records)
        if len(records) < bs:
            records = self._pad_hl_batch(records, bs)
        else:
            self._last_hl_note = None  # Cleared: a later partial batch should re-announce itself.
        if not self._hl_replay_logged_once and (
            self._hl_replay_pool_specs or self.hl_online_bad_fraction >= 0.0
        ):
            self._hl_replay_logged_once = True
            comp = {}
            for r in records:
                comp[r["pool"]] = comp.get(r["pool"], 0) + 1
            msg = f"[steervla.update_hl] HL batch mix (pool -> count): {comp}"
            if online_short > 0:
                msg += (
                    f"; online pool {online_short} short of its share at "
                    f"bad_fraction={self.hl_online_bad_fraction:g} -> backfilled from replay"
                )
            if self.hl_online_bad_fraction >= 0.0:
                n_bad = sum(1 for r in records if r.get("pool") == "online" and self._is_bad_entry(r))
                n_online = sum(1 for r in records if r.get("pool") == "online")
                msg += f"; online label split (BAD/precursor -> {n_bad}, GOOD/null -> {n_online - n_bad})"
                if self.use_adaptive_sampling:
                    cats: dict[str, int] = {}
                    for r in records:
                        if r.get("pool") == "online":
                            c = self._adaptive_category(r)
                            cats[c] = cats.get(c, 0) + 1
                    ordered_cats = {
                        k: cats[k] for k in ADAPTIVE_SAMPLING_WEIGHTS if k in cats
                    }
                    msg += f"; adaptive online categories -> {ordered_cats}"
                elif self.hl_online_precursor_fraction >= 0.0:
                    n_precursor = sum(
                        1
                        for r in records
                        if r.get("pool") == "online" and self._is_bad_entry(r) and self._is_precursor_entry(r)
                    )
                    msg += f"; corrective split (precursor -> {n_precursor}, direct BAD -> {n_bad - n_precursor})"
            print(msg, flush=True)
        return records

    def _log_adaptive_sampling_once(self, entries: list[dict[str, Any]]) -> None:
        """Announce the adaptive-sampling mode, and warn if the corpus carries no outcome tags.

        The warning matters: an online pool written before ``resolve_window_outcome`` existed has
        every ``outcome`` empty, which collapses ``*_catastrophic`` and ``good_success`` to zero
        occurrences. That still trains — it just silently degrades to weighting precursor over
        direct BAD and GOOD over unlabeled — and the difference is invisible without saying so.
        """
        if not self._adaptive_logged_once:
            self._adaptive_logged_once = True
            weights = ", ".join(f"{k}={v:g}" for k, v in self.adaptive_sampling_weights.items())
            print(
                f"[steervla.update_hl] adaptive online sampling ON (weights: {weights}); "
                f"corrective/reinforce coverage still fixed by hl_online_bad_fraction="
                f"{self.hl_online_bad_fraction:g}",
                flush=True,
            )
            if self.hl_online_precursor_fraction >= 0.0:
                print(
                    "[steervla.update_hl] NOTE: hl_online_precursor_fraction="
                    f"{self.hl_online_precursor_fraction:g} is IGNORED under use_adaptive_sampling "
                    "— the precursor/direct balance comes from the sample weights instead.",
                    flush=True,
                )
        if self._adaptive_no_outcomes_warned:
            return
        if any(str(e.get("outcome") or "").strip() for e in entries):
            # A tagged sample has appeared; stop checking.
            self._adaptive_no_outcomes_warned = True
            return
        if len(entries) >= 32:  # enough of a sample to conclude the corpus really has no tags.
            self._adaptive_no_outcomes_warned = True
            print(
                f"[steervla.update_hl] WARNING: use_adaptive_sampling is on but none of "
                f"{len(entries)} online samples carry an 'outcome' tag — the catastrophic/success "
                "categories will never fire. This is expected for a pool collected before "
                "coaches/cast_relabel.py started writing outcomes; re-collect to use them.",
                flush=True,
            )

    def _hl_short_batch(self, got: int, bs: int):
        """Report a batch with nothing readable in it and skip the update."""
        self._hl_skip(
            f"HL pool has {self._hl_pool_size} samples but only {got}/{bs} were readable "
            f"(missing or half-written .npz under {self.hl_dataset_dir}); skipping this update"
        )
        return None

    def _pad_hl_batch(self, records: list[dict[str, Any]], bs: int) -> list[dict[str, Any]]:
        """Repeat-sample ``records`` up to ``bs`` rows so the batch shape stays fixed.

        Only reached when the online pool has started but neither it nor the replay pools can cover a
        full batch yet (typically the first few updates of a run, or replay pools not extracted). The
        alternative — training on a genuinely smaller batch — retraces/recompiles ``_hl_train_step``
        for every new size and re-allocates its backward buffers, so instead the available rows are
        cycled (in shuffled passes) to fill ``bs``. Gradient-wise this is the mean over the distinct
        rows; it just costs the extra compute of the duplicated rows.
        """
        n = len(records)
        if n >= bs or n == 0:
            return records[:bs]
        padded = list(records)
        while len(padded) < bs:
            extra = list(records)
            np.random.shuffle(extra)
            padded.extend(extra[: bs - len(padded)])
        self._hl_note(
            f"partial HL batch: {n}/{bs} distinct samples available "
            f"(online pool {self._hl_pool_size}); repeating them to fill the batch"
        )
        return padded

    def _hl_note(self, msg: str) -> None:
        """Print an HL-update note once per distinct message (same de-dup idea as :meth:`_hl_skip`)."""
        if msg != getattr(self, "_last_hl_note", None):
            print(f"[steervla.update_hl] {msg}", flush=True)
            self._last_hl_note = msg

    def _record_hl_sample_uses(self, records: list[dict[str, Any]], num_grad_steps: int) -> dict[str, float]:
        """Book one HL update against the per-sample use counters and return reuse telemetry.

        ``records`` is the batch about to be fed to ``num_grad_steps`` optimizer steps, so every row
        is credited ``num_grad_steps`` uses (and a row that ``_pad_hl_batch`` duplicated is credited
        once per copy — the duplicate really is extra gradient exposure for that sample).

        Two different "average reuse" numbers are reported because they answer different questions
        about overfitting:

        * ``reuse_mean`` — uses per sample **that has ever been used**. This is the re-exposure
          factor of the data the policy has actually seen.
        * ``reuse_online_per_pool_sample`` — total online uses divided by the *current online pool
          size*, i.e. counting never-drawn samples as zero. This is the one that tracks the small
          cast_relabel pool being ground over and over as the run goes on.
        """
        ns = max(1, int(num_grad_steps))
        for r in records:
            sid = r.get("sample_id")
            if not sid:
                continue
            pool = str(r.get("pool", "online"))
            self._hl_sample_uses[sid] = self._hl_sample_uses.get(sid, 0) + ns
            per_pool = self._hl_sample_uses_by_pool.setdefault(pool, {})
            per_pool[sid] = per_pool.get(sid, 0) + ns

        out: dict[str, float] = {}
        uses = list(self._hl_sample_uses.values())
        if uses:
            total = float(sum(uses))
            out["reuse_total_uses"] = total
            out["reuse_distinct_samples"] = float(len(uses))
            out["reuse_mean"] = total / len(uses)
            out["reuse_max"] = float(max(uses))
        online = self._hl_sample_uses_by_pool.get("online", {})
        if online:
            online_uses = list(online.values())
            online_total = float(sum(online_uses))
            out["reuse_online_total_uses"] = online_total
            out["reuse_online_distinct_samples"] = float(len(online_uses))
            out["reuse_online_mean"] = online_total / len(online_uses)
            out["reuse_online_max"] = float(max(online_uses))
            pool_size = int(self._hl_pool_size)
            if pool_size > 0:
                # Counts never-drawn pool samples as zero, so this only rises once the pool stops
                # growing as fast as it is consumed.
                out["reuse_online_per_pool_sample"] = online_total / float(pool_size)
                out["reuse_online_pool_coverage"] = min(1.0, len(online_uses) / float(pool_size))
        out["online_pool_size"] = float(self._hl_pool_size)
        return out

    def _hl_skip(self, reason: str) -> dict[str, float]:
        """No-op the HL update, printing *why* — but only when the reason changes.

        ``update_hl`` runs on every ``update_with_vla`` call, so an unconditional print would
        spam the log. Printing on transitions keeps one line per distinct cause (and re-prints
        if the run regresses into a previously-cleared state).
        """
        if reason != self._last_hl_skip_reason:
            print(f"[steervla.update_hl] no HL update: {reason}", flush=True)
            self._last_hl_skip_reason = reason
        return {}
