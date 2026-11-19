#!/usr/bin/env bash
# Kept for the sweep driver, which execs this name: qwen_zs_bon_run.sh now takes RENDER_GPU itself.
exec "$(dirname "${BASH_SOURCE[0]}")/qwen_zs_bon_run.sh" "$@"
