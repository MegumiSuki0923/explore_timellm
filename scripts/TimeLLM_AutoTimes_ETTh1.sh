#!/usr/bin/env bash
set -euo pipefail

dataset=ETTh1
seq_len=512
comment=${1:-ar_direct}
timestamp=$(date +"%Y-%m-%d_%H:%M")
log_dir="./logs/${dataset}"
log_file="${log_dir}/${dataset}_${seq_len}_${comment}_${timestamp}.log"
mkdir -p "$log_dir"

extra_args=()
if [[ "$comment" != ar_direct && "$comment" != ar_* ]]; then
  echo "variant must be ar_direct or an ar_* module combination: $comment" >&2
  exit 2
fi
if [[ "$comment" == *timestamp* ]]; then
  extra_args+=(--use_timestamp --timestamp_cache ./dataset/ETT-small/ETTh1_gpt2_tl64.pt)
fi
if [[ "$comment" == *reprogram* ]]; then
  extra_args+=(--use_reprogram)
fi
if [[ "$comment" == *prompt* ]]; then
  extra_args+=(--use_prompt)
fi

python -u run_autotimes.py \
  --data "$dataset" \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --features M \
  --seq_len 512 \
  --label_len 448 \
  --token_len 64 \
  --batch_size 24 \
  --learning_rate 0.0005 \
  --train_epochs 10 \
  --llm_model GPT2 \
  --llm_layers 12 \
  --llm_dim 768 \
  --seed 2021 \
  --model_comment "$comment" \
  "${extra_args[@]}" \
  2>&1 | tee -a "$log_file"

python summarize_autotimes_log.py "$log_file" | tee -a "$log_file"
