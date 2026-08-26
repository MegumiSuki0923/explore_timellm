model_name=TimeLLM
train_epochs=100
learning_rate=0.01
llm_model='GPT2'
gpt2_layers=12
llm_dim=768

master_port=00097
num_process=1
batch_size=24
d_model=32
d_ff=128

dataset=ETTh1
seq_len=512
comment='gpt2'

# dynamically generate log file path
timestamp=$(date +"%Y-%m-%d_%H:%M")
log_dir="./logs/${dataset}"
mkdir -p "$log_dir"
log_file="${log_dir}/${dataset}_${seq_len}_${comment}_${timestamp}.log"

{
echo "=========================================="
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Start running:"
echo "  dataset      : ${dataset}"
echo "  model        : ${model_name}"
echo "  llm_model    : ${llm_model}"
echo "  seq_len      : ${seq_len}"
echo "  pred_len     : 96"
echo "  learning_rate: ${learning_rate}"
echo "  batch_size   : ${batch_size}"
echo "=========================================="
} | tee -a "$log_file"
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
  --llm_layers $gpt2_layers \
  --llm_dim $llm_dim \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --save_checkpoint 0 \
  --patience 3 \
  2>&1 | tee -a "$log_file"

{
echo "=========================================="
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Start running:"
echo "  dataset      : ${dataset}"
echo "  model        : ${model_name}"
echo "  llm_model    : ${llm_model}"
echo "  seq_len      : ${seq_len}"
echo "  pred_len     : 192"
echo "  learning_rate: 0.02"
echo "  batch_size   : ${batch_size}"
echo "=========================================="
} | tee -a "$log_file"
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
  --d_model 32 \
  --d_ff 128 \
  --batch_size $batch_size \
  --learning_rate 0.02 \
  --llm_model $llm_model \
  --llm_layers $gpt2_layers \
  --llm_dim $llm_dim \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --save_checkpoint 0 \
  --patience 3 \
  2>&1 | tee -a "$log_file"

{
echo "=========================================="
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Start running:"
echo "  dataset      : ${dataset}"
echo "  model        : ${model_name}"
echo "  llm_model    : ${llm_model}"
echo "  seq_len      : ${seq_len}"
echo "  pred_len     : 336"
echo "  learning_rate: 0.001"
echo "  batch_size   : ${batch_size}"
echo "=========================================="
} | tee -a "$log_file"
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
  --lradj 'COS'\
  --learning_rate 0.001 \
  --llm_model $llm_model \
  --llm_layers $gpt2_layers \
  --llm_dim $llm_dim \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --save_checkpoint 0 \
  --patience 3 \
  2>&1 | tee -a "$log_file"

{
echo "=========================================="
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Start running:"
echo "  dataset      : ${dataset}"
echo "  model        : ${model_name}"
echo "  llm_model    : ${llm_model}"
echo "  seq_len      : ${seq_len}"
echo "  pred_len     : 720"
echo "  learning_rate: ${learning_rate}"
echo "  batch_size   : ${batch_size}"
echo "=========================================="
} | tee -a "$log_file"
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
  --llm_layers $gpt2_layers \
  --llm_dim $llm_dim \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --save_checkpoint 0 \
  --patience 3 \
  2>&1 | tee -a "$log_file"
