"""Live HTML viewer for the CAST-relabel VLM transcripts.

Each reviewed window writes ``vlm_calls.json`` next to its ``rollout.mp4`` (see
``OnlineCastRelabelSession._run_window``): every prompt sent and every raw response received,
for the window review and the credit/relabel call. Those are built on the fly and otherwise
never touch disk, so without this there is no way to read what the VLM actually said.

This renders them into one page that is regenerated after every window, so a browser tab left
open on it follows the run (the page meta-refreshes). Unlike ``cast_relabel_viewer.py`` -- the
offline deep-dive, which base64-embeds videos and telemetry and is far too heavy to rebuild
per window -- this one links videos by relative path and carries only text, so a refresh is
milliseconds regardless of how many windows have accumulated.

With ``cast_relabel.two_stage_review`` on, the review is two calls and both are shown: Step 1
(scene + traffic-flow state, as prose) and Step 2 (the GOOD/BAD events derived from it). With it
off there is a single review prompt and only the parsed events.

Usage (standalone, on a finished or running run)::

    .venv/bin/python impls/coaches/cast_review_viewer.py <run-dir-or-cast_relabel-dir>
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

VIEWER_NAME = "review_viewer.html"
DEFAULT_REFRESH_SEC = 30


def find_cast_dir(path: Path) -> Path:
    """Accept a run dir, its ``cast_relabel/`` dir, or a single window dir."""
    path = Path(path)
    if (path / "vlm_calls.json").is_file() or (path / "cast_relabel.json").is_file():
        return path.parent
    if path.name == "cast_relabel":
        return path
    cand = path / "cast_relabel"
    if cand.is_dir():
        return cand
    return path


def collect_windows(cast_dir: Path) -> list[dict[str, Any]]:
    """Load every window transcript under ``cast_dir``, oldest first.

    Windows still being written are skipped rather than shown half-parsed: the review worker
    writes ``vlm_calls.json`` in one shot, but a reader can still catch a partial file.
    """
    out: list[dict[str, Any]] = []
    for wdir in sorted(p for p in Path(cast_dir).iterdir() if p.is_dir()):
        calls_path = wdir / "vlm_calls.json"
        if not calls_path.is_file():
            continue
        try:
            calls = json.loads(calls_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        cast_json: dict[str, Any] = {}
        cj = wdir / "cast_relabel.json"
        if cj.is_file():
            try:
                cast_json = json.loads(cj.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                cast_json = {}
        video = calls.get("video") or "rollout.mp4"
        out.append(
            {
                "dir": wdir.name,
                "video_rel": f"{wdir.name}/{video}" if (wdir / video).is_file() else "",
                "calls": calls,
                "chunks": cast_json.get("action_chunks", []),
                "mtime": calls_path.stat().st_mtime,
            }
        )
    return out


def _esc(v: Any) -> str:
    return html.escape("" if v is None else str(v))


def _pre(text: Any, *, cls: str = "") -> str:
    body = _esc(text).strip()
    if not body:
        return '<p class="empty">(empty)</p>'
    return f'<pre class="{cls}">{body}</pre>'


def _details(summary: str, inner: str, *, open_: bool = False) -> str:
    return (
        f'<details{" open" if open_ else ""}><summary>{_esc(summary)}</summary>{inner}</details>'
    )


def _events_html(events: list[dict[str, Any]]) -> str:
    if not events:
        return '<p class="empty">(no events returned)</p>'
    rows = []
    for e in events:
        lab = str(e.get("label") or "")
        rows.append(
            f'<tr class="{"bad" if lab.upper()=="BAD" else "good"}">'
            f'<td class="ts">{_esc(e.get("timestamp_sec"))}s</td>'
            f'<td class="lab">{_esc(lab)}</td>'
            f'<td>{_esc(e.get("description"))}'
            + (
                f'<div class="corr"><b>correction:</b> {_esc(e.get("correction"))}</div>'
                if str(e.get("correction") or "").strip()
                else ""
            )
            + "</td></tr>"
        )
    return (
        '<table class="ev"><thead><tr><th>t</th><th>label</th><th>description</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _chunks_html(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return '<p class="empty">(no chunk credits)</p>'
    rows = []
    for c in chunks:
        lab = str(c.get("label") or "")
        subs = c.get("suggested_subtasks") or []
        rows.append(
            f'<tr class="{"bad" if lab.upper()=="BAD" else ("good" if lab.upper()=="GOOD" else "")}">'
            f'<td>{_esc(c.get("chunk_index"))}</td>'
            f'<td class="lab">{_esc(lab) or "-"}</td>'
            f'<td>{_esc(c.get("credit_source"))}</td>'
            f'<td>{_esc(c.get("original_subtask"))}</td>'
            f'<td>{_esc("; ".join(str(x) for x in subs))}</td>'
            f'<td>{_esc(c.get("rationale"))}</td>'
            f'<td>{_esc(c.get("suggested_reasoning"))}</td></tr>'
        )
    return (
        '<table class="ev"><thead><tr><th>#</th><th>label</th><th>src</th><th>original subtask</th>'
        "<th>suggested subtask</th><th>rationale</th><th>suggested reasoning</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _window_html(w: dict[str, Any], idx: int) -> str:
    c = w["calls"]
    two = bool(c.get("two_stage"))
    vid = (
        f'<video class="vid" controls preload="none" src="{_esc(w["video_rel"])}"></video>'
        if w["video_rel"]
        else '<p class="empty">(video not found)</p>'
    )
    parts = [
        f'<section class="win" id="w{idx}">',
        f'<h2>{_esc(w["dir"])} '
        f'<span class="sub">ep {_esc(c.get("episode"))} · window {_esc(c.get("window_index"))} · '
        f'{"TWO-STAGE" if two else "single-call"} · {_esc(c.get("route"))}</span></h2>',
        vid,
    ]
    if two:
        parts.append(
            '<h3 class="s1">Step 1 — scene &amp; traffic-flow analysis <span class="sub">'
            "(call 1 of 2, answered as prose; carried verbatim into call 2)</span></h3>"
        )
        parts.append(_pre(c.get("stage1_response") or c.get("review_stage1_response"), cls="resp"))
        parts.append(
            _details("Step 1 prompt", _pre(c.get("review_stage1_prompt"), cls="prompt"))
        )
        parts.append(
            '<h3 class="s2">Step 2 — behaviour review <span class="sub">'
            "(call 2 of 2, given the Step 1 answer as established context)</span></h3>"
        )
        parts.append(_events_html(c.get("events") or []))
        parts.append(
            _details("Step 2 raw response", _pre(c.get("review_stage2_response"), cls="resp"))
        )
        parts.append(
            _details("Step 2 prompt", _pre(c.get("review_stage2_prompt"), cls="prompt"))
        )
    else:
        parts.append(
            '<h3 class="s2">Review events <span class="sub">(single call: Steps 1 and 2 together; '
            "the Step 1 reasoning is internal and not recoverable — set "
            "cast_relabel.two_stage_review=True to capture it)</span></h3>"
        )
        parts.append(_events_html(c.get("events") or []))
        parts.append(_details("Review prompt", _pre(c.get("review_prompt"), cls="prompt")))

    # Route divergence: shown between the review and the credit pass because that is exactly
    # where it acts -- a diverged window has already had its post-divergence chunks removed
    # before the credit prompt below was built, so the chunk list is short on purpose.
    div = c.get("route_divergence") or {}
    if div.get("diverged"):
        ts = div.get("timestamp_sec")
        ts_txt = f"t={float(ts):.2f}s" if ts is not None else "unlocated"
        dropped = c.get("route_divergence_dropped_chunks")
        parts.append(
            '<h3 class="s2">Route divergence <span class="sub">'
            f"(ego left the planned route at {_esc(ts_txt)}"
            + (f"; {int(dropped)} chunk(s) dropped before credit assignment" if dropped else "")
            + ")</span></h3>"
        )
        parts.append(
            '<div class="diverged"><b>DIVERGED &mdash; supervision cut here.</b> '
            + _esc(str(div.get("reason") or "(no reason given)"))
            + "</div>"
        )

    parts.append('<h3 class="s3">Credit assignment &amp; relabel</h3>')
    parts.append(_chunks_html(w.get("chunks") or []))
    parts.append(_details("Credit raw response", _pre(c.get("credit_response"), cls="resp")))
    parts.append(_details("Credit prompt", _pre(c.get("credit_prompt"), cls="prompt")))
    parts.append("</section>")
    return "".join(parts)


_CSS = """
:root{--bg:#0f1115;--fg:#e6e6e6;--dim:#9aa0a6;--card:#171a21;--line:#2a2f3a;--good:#2e7d32;--bad:#b3261e}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:13px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{position:sticky;top:0;z-index:5;background:#0b0d11;border-bottom:1px solid var(--line);padding:10px 16px}
header h1{margin:0;font-size:15px}
header .meta{color:var(--dim);font-size:12px;margin-top:2px}
.wrap{display:flex;gap:0;align-items:flex-start}
nav{position:sticky;top:52px;width:230px;flex:0 0 230px;max-height:calc(100vh - 52px);overflow:auto;border-right:1px solid var(--line);padding:10px}
nav a{display:block;color:var(--dim);text-decoration:none;padding:4px 6px;border-radius:4px;font-size:12px}
nav a:hover{background:var(--card);color:var(--fg)}
main{flex:1;min-width:0;padding:16px 20px}
.win{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px 16px;margin-bottom:22px}
.win h2{margin:0 0 10px;font-size:14px}
.sub{color:var(--dim);font-weight:400;font-size:12px}
h3{margin:16px 0 6px;font-size:13px;border-left:3px solid var(--line);padding-left:8px}
h3.s1{border-left-color:#3b82f6}h3.s2{border-left-color:#a855f7}h3.s3{border-left-color:#f59e0b}
.vid{width:100%;max-width:620px;border-radius:6px;background:#000;display:block}
pre{white-space:pre-wrap;word-wrap:break-word;background:#0b0d11;border:1px solid var(--line);
border-radius:6px;padding:10px;margin:6px 0;max-height:520px;overflow:auto;font-size:12px}
pre.prompt{color:var(--dim)}
.diverged{background:#3a1414;border-left:3px solid #d9534f;padding:8px 10px;margin:6px 0;border-radius:3px}
details{margin:6px 0}summary{cursor:pointer;color:var(--dim);font-size:12px;user-select:none}
summary:hover{color:var(--fg)}
table.ev{width:100%;border-collapse:collapse;margin:6px 0;font-size:12px}
table.ev th,table.ev td{border:1px solid var(--line);padding:5px 7px;vertical-align:top;text-align:left}
table.ev th{background:#0b0d11;color:var(--dim);font-weight:600}
tr.good .lab{color:#7ddb85}tr.bad .lab{color:#ff8a80}
td.ts{white-space:nowrap;color:var(--dim)}
.corr{margin-top:4px;color:#ffcc80}
.empty{color:var(--dim);font-style:italic;margin:6px 0}
"""

_JS = """
// Preserve scroll position and which <details> are open across the meta-refresh, so a tab left
// open on a live run does not jump to the top every 30s.
(function(){
  var K='castviewer:'+location.pathname;
  try{
    var st=JSON.parse(sessionStorage.getItem(K)||'{}');
    if(st.y) window.scrollTo(0,st.y);
    (st.open||[]).forEach(function(i){var d=document.querySelectorAll('details')[i]; if(d) d.open=true;});
  }catch(e){}
  function save(){
    try{
      var open=[];document.querySelectorAll('details').forEach(function(d,i){if(d.open)open.push(i);});
      sessionStorage.setItem(K,JSON.stringify({y:window.scrollY,open:open}));
    }catch(e){}
  }
  window.addEventListener('beforeunload',save);
  setInterval(save,2000);
})();
"""


def render(windows: list[dict[str, Any]], *, title: str, refresh_sec: int = DEFAULT_REFRESH_SEC) -> str:
    import datetime

    nav = "".join(
        f'<a href="#w{i}">{_esc(w["dir"])}'
        + (" <b>·2</b>" if w["calls"].get("two_stage") else "")
        + "</a>"
        for i, w in enumerate(windows)
    )
    body = "".join(_window_html(w, i) for i, w in enumerate(windows))
    if not windows:
        body = '<p class="empty">No reviewed windows yet. This page refreshes automatically.</p>'
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    two_n = sum(1 for w in windows if w["calls"].get("two_stage"))
    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="{int(refresh_sec)}">
<title>{_esc(title)}</title><style>{_CSS}</style></head>
<body>
<header>
  <h1>CAST relabel — VLM review transcripts</h1>
  <div class="meta">{_esc(title)} · {len(windows)} window(s), {two_n} two-stage ·
  regenerated {stamp} · auto-refresh {int(refresh_sec)}s</div>
</header>
<div class="wrap"><nav>{nav}</nav><main>{body}</main></div>
<script>{_JS}</script>
</body></html>
"""


def refresh(cast_dir: Path, *, title: str = "", out_path: Path | None = None,
            refresh_sec: int = DEFAULT_REFRESH_SEC) -> Path:
    """Regenerate the viewer for one run. Returns the written path."""
    cast_dir = Path(cast_dir)
    windows = collect_windows(cast_dir)
    out = Path(out_path) if out_path is not None else cast_dir / VIEWER_NAME
    out.write_text(
        render(windows, title=title or str(cast_dir), refresh_sec=refresh_sec), encoding="utf-8"
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", type=Path, help="run dir, cast_relabel/ dir, or a window dir")
    ap.add_argument("-o", "--out", type=Path, default=None)
    ap.add_argument("--refresh-sec", type=int, default=DEFAULT_REFRESH_SEC)
    a = ap.parse_args()
    cast_dir = find_cast_dir(a.path)
    out = refresh(cast_dir, title=str(cast_dir), out_path=a.out, refresh_sec=a.refresh_sec)
    n = len(collect_windows(cast_dir))
    print(f"wrote {out} ({n} window(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
