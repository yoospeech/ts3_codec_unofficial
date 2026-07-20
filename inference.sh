#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

: "${CKPT:?Set CKPT to the downloaded Hugging Face checkpoint path}"
: "${INPUT_DIR:?Set INPUT_DIR to a directory containing WAV files}"
: "${OUTPUT_DIR:=$ROOT_DIR/recon_wavs}"

python inference.py \
  "ckpt=$CKPT" \
  "input_dir=$INPUT_DIR" \
  "output_dir=$OUTPUT_DIR" \
  "hydra.output_subdir=null" \
  "hydra.job.chdir=False"
