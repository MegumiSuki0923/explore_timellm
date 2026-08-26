model_name=TimeLLM
train_epochs=100
learning_rate=0.01
llm_model='MAMBA'
llm_layers=24
llm_dim=768

master_port=00096
num_process=1
batch_size=24
d_model=32
d_ff=128

dataset=ETTh1
seq_len=512
comment='mamba130m'

# dynamically generate log file path
timestamp=$(date +"%Y-%m-%d_%H:%M")
log_dir="./logs/${dataset}"
mkdir -p "$log_dir"
log_file="${log_dir}/${dataset}_${seq_len}_${comment}_${timestamp}.log"

print_run_info() {
  local pred_len=$1
  local learning_rate=$2
  echo "==========================================" | tee -a "$log_file"
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] Start running:" | tee -a "$log_file"
  echo "  dataset      : $dataset" | tee -a "$log_file"
  echo "  model        : $model_name" | tee -a "$log_file"
  echo "  llm_model    : $llm_model" | tee -a "$log_file"
  echo "  seq_len      : $seq_len" | tee -a "$log_file"
  echo "  pred_len     : $pred_len" | tee -a "$log_file"
  echo "  learning_rate: $learning_rate" | tee -a "$log_file"
  echo "  batch_size   : $batch_size" | tee -a "$log_file"
  echo "==========================================" | tee -a "$log_file"
}

print_run_info 96 $learning_rate
accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_512_96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 96 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_model $llm_model \
  --llm_layers $llm_layers \
  --llm_dim $llm_dim \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --save_checkpoint 0 \
  2>&1 | tee -a "$log_file"

print_run_info 192 0.02
accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_512_192 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 192 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate 0.02 \
  --llm_model $llm_model \
  --llm_layers $llm_layers \
  --llm_dim $llm_dim \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --save_checkpoint 0 \
  2>&1 | tee -a "$log_file"

print_run_info 336 0.001
accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_512_336 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 336 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --lradj 'type1' \
  --learning_rate 0.001 \
  --llm_model $llm_model \
  --llm_layers $llm_layers \
  --llm_dim $llm_dim \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --save_checkpoint 0 \
  2>&1 | tee -a "$log_file"

print_run_info 720 $learning_rate
accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_512_720 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 720 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_model $llm_model \
  --llm_layers $llm_layers \
  --llm_dim $llm_dim \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --save_checkpoint 0 \
  2>&1 | tee -a "$log_file"
