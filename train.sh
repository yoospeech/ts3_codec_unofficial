#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

: "${LIBRISPEECH_ROOT:=}"
: "${DEVICES:=1}"
: "${ACCELERATOR:=gpu}"
: "${PRECISION:=bf16}"
: "${MAX_STEPS:=500000}"
: "${BATCH_SIZE:=64}"
: "${LOG_DIR:=pl_log}"
: "${TRAIN_FILELIST:=./filelists/librispeech_train.txt}"
: "${VAL_FILELIST:=$TRAIN_FILELIST}"
: "${TEST_FILELIST:=./filelists/librispeech_test.txt}"

ROOT_OVERRIDE_ARGS=()
if [[ -n "$LIBRISPEECH_ROOT" ]]; then
  ROOT_OVERRIDE_ARGS+=("preprocess.datasets.LibriSpeech.root=$LIBRISPEECH_ROOT")
fi

python train.py \
  "${ROOT_OVERRIDE_ARGS[@]}" \
  preprocess.audio.sr=16000 \
  preprocess.stft.hop_length=320 \
  dataset.frame_length=320 \
  dataset.min_audio_length=160000 \
  dataset.train.filelist="$TRAIN_FILELIST" \
  dataset.val.filelist="$VAL_FILELIST" \
  dataset.test.filelist="$TEST_FILELIST" \
  dataset.train.batch_size="$BATCH_SIZE" \
  dataset.val.batch_size="$BATCH_SIZE" \
  model.codec_encoder.transformer_only=True \
  model.codec_decoder.transformer_only=True \
  train.trainer.accelerator="$ACCELERATOR" \
  train.trainer.devices="$DEVICES" \
  train.trainer.precision="$PRECISION" \
  train.trainer.max_steps="$MAX_STEPS" \
  train.trainer.min_steps="$MAX_STEPS" \
  log_dir="$LOG_DIR" \
  hydra.output_subdir=null \
  hydra.job.chdir=False \
  "$@"
