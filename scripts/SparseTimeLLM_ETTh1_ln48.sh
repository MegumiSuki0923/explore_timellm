# SparseTimeLLM ETTh1 — ln48 config (LayerNorm + 48-point blocks), all 4 horizons.
# Hyperparameters mirror scripts/SparseTimeLLM_ETTh1.sh per-horizon values exactly.

model_name=SparseTimeLLM
train_epochs=100
llm_model='GPT2'
gpt2_layers=12
llm_dim=768
period_len=24

master_port=00098
num_process=1
batch_size=8
llm_chunk_size=56
d_model=32
d_ff=128

dataset=ETTh1
seq_len=512
comment='sparse_v3_ln48'

timestamp=$(date +"%Y-%m-%d_%H:%M")
log_dir="./logs/${dataset}"
mkdir -p "$log_dir"
log_file="${log_dir}/${dataset}_${seq_len}_${comment}_${timestamp}.log"

for pred_len_arg in 96 192 336 720; do
  case $pred_len_arg in
    96)  lr=0.01;  lradj_flag="" ;;
    192) lr=0.02;  lradj_flag="" ;;
    336) lr=0.001; lradj_flag="--lradj COS" ;;
    720) lr=0.01;  lradj_flag="" ;;
  esac

  {
  echo "=========================================="
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] Start running:"
  echo "  dataset      : ${dataset}"
  echo "  model        : ${model_name} (ln48)"
  echo "  seq_len      : ${seq_len}"
  echo "  pred_len     : ${pred_len_arg}"
  echo "  period_len   : ${period_len}"
  echo "  learning_rate: ${lr} ${lradj_flag}"
  echo "=========================================="
  } | tee -a "$log_file"

  accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_512_${pred_len_arg} \
    --model $model_name \
    --data ETTh1 \
    --features M \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len_arg \
    --period_len $period_len \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 1 \
    --d_model $d_model \
    --d_ff $d_ff \
    --batch_size $batch_size \
    --llm_chunk_size $llm_chunk_size \
    --learning_rate $lr \
    $lradj_flag \
    --llm_model $llm_model \
    --llm_layers $gpt2_layers \
    --llm_dim $llm_dim \
    --train_epochs $train_epochs \
    --model_comment $comment \
    --save_checkpoint 0 \
    --patience 3 \
    2>&1 | tee -a "$log_file"
done
