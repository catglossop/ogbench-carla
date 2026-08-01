"""Assemble the CHOSEN best-of-N candidate frame at each step into an mp4.

Reads <candidates_dir>/step{N:06d}_gemini.txt (same "choice=" line format works
for a critic run's saved text too) to find which candidate index was selected
each step, then stitches step{N:06d}_cand{choice}.jpg frames together in order
via ffmpeg -- i.e. what the agent actually saw and picked, one frame per env
step (not the periodic/panel-only "bon/candidates" snapshots).

Uses ffmpeg (via the imageio_ffmpeg-bundled binary) rather than cv2.VideoWriter,
which can produce mp4 containers some players fail to decode.

Usage:
  python impls/assemble_chosen_candidates_video.py \
      --candidates_dir /path/to/videos/candidates \
      --out /path/to/videos/chosen_candidates.mp4 --fps 10
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_string("candidates_dir", None, "Directory with step*_cand*.jpg + step*_gemini.txt.", required=True)
flags.DEFINE_string("out", None, "Output mp4 path. Defaults to <candidates_dir>/chosen_candidates.mp4.")
flags.DEFINE_float("fps", 10.0, "Output video fps.")

_STEP_RE = re.compile(r"^step(\d{6})_gemini\.txt$")


def _ffmpeg_binary() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def main(_argv):
    import cv2  # type: ignore -- only for burning the step/choice caption + reading frame size

    cand_dir = Path(FLAGS.candidates_dir)
    out_path = Path(FLAGS.out) if FLAGS.out else cand_dir / "chosen_candidates.mp4"

    steps: list[tuple[int, int]] = []  # (step, choice)
    for f in sorted(cand_dir.glob("step*_gemini.txt")):
        m = _STEP_RE.match(f.name)
        if not m:
            continue
        step = int(m.group(1))
        choice = None
        for line in f.read_text().splitlines():
            if line.startswith("choice="):
                choice = int(line[len("choice="):].strip())
                break
        if choice is None:
            print(f"[assemble] {f.name}: no choice= line, skipping")
            continue
        frame_path = cand_dir / f"step{step:06d}_cand{choice}.jpg"
        if not frame_path.exists():
            print(f"[assemble] {frame_path.name} missing, skipping step {step}")
            continue
        steps.append((step, choice))

    if not steps:
        raise SystemExit(f"No usable step*_gemini.txt + matching frame found in {cand_dir}")

    steps.sort(key=lambda x: x[0])

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for seq, (step, choice) in enumerate(steps):
            frame = cv2.imread(str(cand_dir / f"step{step:06d}_cand{choice}.jpg"))
            if frame is None:
                continue
            h = frame.shape[0]
            cv2.putText(
                frame, f"step {step}  chosen={choice}", (4, h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA,
            )
            cv2.imwrite(str(tmp / f"{seq:06d}.jpg"), frame)

        ffmpeg = _ffmpeg_binary()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            ffmpeg, "-y",
            "-framerate", str(FLAGS.fps),
            "-i", str(tmp / "%06d.jpg"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr)
            raise SystemExit(f"ffmpeg failed with exit code {result.returncode}")

    print(f"[assemble] wrote {len(steps)} frames -> {out_path}")


if __name__ == "__main__":
    app.run(main)
