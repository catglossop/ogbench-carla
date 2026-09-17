"""Standalone checks for the HL batch wandb tables (no model, no CARLA, no wandb).

    JAX_PLATFORMS=cpu PYTHONPATH=impls uv run python impls/vlas/test_hl_batch_logging.py

Covers the 2026-09-09 addition of ``source_pool`` -- which ADAPTIVE_SAMPLING_WEIGHTS bucket each
sample was drawn from -- to ``vla_hl/batch_text`` and ``vla_hl/batch_tokens``. That is a different
axis from the existing ``pool`` column, which says which REPLAY pool a sample came from.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FAILURES: list[str] = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  -- ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


from vlas.steervla import ADAPTIVE_SAMPLING_WEIGHTS, SteerVLAActor

cat = SteerVLAActor._adaptive_category

# ── 1. every bucket the sampler weights is reachable, and resolution is total ──────────
print("\n[1] _adaptive_category covers the weight table")
cases = {
    "bad_precursor_catastrophic": {"label": "BAD", "credit_source": "precursor", "outcome": "collision"},
    "bad_precursor": {"label": "BAD", "credit_source": "precursor", "outcome": ""},
    "bad_direct_catastrophic": {"label": "BAD", "credit_source": "direct", "outcome": "collision"},
    "bad_direct": {"label": "BAD", "credit_source": "direct", "outcome": ""},
    "good_success": {"label": "GOOD", "outcome": "route_completed"},
    "good": {"label": "GOOD", "outcome": ""},
    "unlabeled": {"label": None, "outcome": ""},
}
for expected, entry in cases.items():
    got = cat(entry)
    check(f"{expected}", got == expected, f"got {got}")
check(
    "every category has a weight",
    all(c in ADAPTIVE_SAMPLING_WEIGHTS for c in cases),
    str(sorted(set(cases) - set(ADAPTIVE_SAMPLING_WEIGHTS))),
)
check("an empty entry still resolves", cat({}) == "unlabeled", cat({}))

# ── 2. the tables carry it, and their widths still line up ────────────────────────────
print("\n[2] table shapes")
src = Path("impls/vlas/steervla.py").read_text()

text_cols = re.search(
    r'wandb\.Table\(\s*columns=\[\s*(.*?)\]\s*\)', src, re.DOTALL
).group(1)
text_col_names = re.findall(r'"([a-z_]+)"', text_cols)
check("batch_text has a source_pool column", "source_pool" in text_col_names, str(text_col_names))
check("batch_text still has pool (a different axis)", "pool" in text_col_names)

# the row literal that feeds that table
row_block = src[src.index("            rows = [\n"): src.index("                for i in range(len(records))")]
# Count the elements of the inner row literal. Every element is its own line ending in a comma;
# do NOT filter on "[" -- most of them are subscripts like pools[i].
_inner = row_block[row_block.index("[\n", row_block.index("rows = [")) :]
n_row_fields = len([ln for ln in _inner.split("\n") if ln.strip().endswith(",")])
check(
    "batch_text row width matches its column count",
    n_row_fields == len(text_col_names),
    f"row fields={n_row_fields} columns={len(text_col_names)}",
)

tok_block = src[src.index('                "hl_update_call",\n                "sample",\n                "pool",'):]
tok_cols = re.findall(r'"([a-z_]+)"', tok_block[: tok_block.index("]")])
check("batch_tokens has a source_pool column", "source_pool" in tok_cols, str(tok_cols[:6]))
check("source_pool sits next to pool in batch_tokens", tok_cols.index("source_pool") == tok_cols.index("pool") + 1)
check(
    "token rows populate source_pool",
    '"source_pool": str(self._adaptive_category(rec))' in src,
)
check(
    "batch_tokens builds rows FROM the column list (so the new column is picked up)",
    "[r.get(c) for c in columns[1:]]" in src,
)

# ── 3. per-bucket scalars, so it shows up in Charts and not just the table ─────────────
print("\n[3] per-bucket scalars")
check("source_pool counts are logged", 'payload[f"vla_hl/source_pool/{sp}"]' in src)
check("the existing pool counts are untouched", 'payload[f"vla_hl/pool/{p}"]' in src)

print()
if FAILURES:
    print(f"FAILED ({len(FAILURES)}): " + ", ".join(FAILURES))
    raise SystemExit(1)
print("all checks passed")
