# Codex Loop：TimeLLM × AutoTimes 贪心消融实验

## 你的任务

在仓库 `/home/Lain/Code/explore_timellm-AutoTimes` 中，使用 `conda` 环境 `time-llm`，依次完成 TimeLLM-AutoTimes 的训练、监控、结果解析和贪心模块晋级，直到得到最终最佳组合或满足停止条件。

必须持续推进，不要只给操作建议。训练耗时较长是正常现象；等待期间监控进程、GPU和日志，不要重复启动同一实验。

## 不可改变的实验配置

```text
dataset      = ETTh1
seq_len      = 512
label_len    = 448
token_len    = 64
batch_size   = 24
learning_rate= 5e-4
GPT-2 layers = 12
seed         = 2021
epochs       = 100（最大值）
patience     = 3
T_max        = 100
checkpoint   = 禁止保存
```

只允许使用 GPT-2，不得改用其他语言模型。不得搜索或微调学习率，不得改变 batch size、序列长度、token长度、epoch数、随机种子或指标口径。

## 正确基线

`baseline.md` 是正确的原始 TimeLLM-GPT2 基线：

| pred_len | MSE | MAE |
|---:|---:|---:|
| 96 | 0.3840 | 0.4124 |
| 192 | 0.4296 | 0.4416 |
| 336 | 0.4730 | 0.4723 |
| 720 | 0.4501 | 0.4703 |
| Avg | **0.4342** | **0.4491** |

保护以下文件，不得修改、覆盖、还原或格式化：

```text
models/TimeLLM.py
layers/Embed.py
baseline.md
```

开始前及每次修复代码后检查哈希：

```text
models/TimeLLM.py  1b0ba5df405d95d6ba33680842446351e63b4553a659355879fb088a1c92a22e
layers/Embed.py     dfecbc22eedb4b266cdf39ba0832711ee46911d170aa1997d234a183ec8f5002
baseline.md         5143fa1871a49a3bd7b9e3370e36e714bde31ca1061ea3582bed99439724f1fe
```

工作树中已有修改属于用户。不得使用 `git reset`、`git checkout --`、`git stash`、清理未跟踪文件或其他会覆盖用户工作的命令。

## 结果比较规则

每个实验结束后运行：

```bash
python summarize_autotimes_log.py <本轮日志路径>
```

日志中每个 horizon 选择最低 Test MSE 的 epoch，并取同一 epoch 的 MAE。四个 horizon 的平均值为 `Avg MSE` 和 `Avg MAE`。

候选模型相对当前 best 的晋级规则：

1. 若候选 `Avg MSE <= best Avg MSE - 0.001`，候选晋级。
2. 若二者 `Avg MSE` 差值绝对值 `< 0.001`，仅当候选 `Avg MAE < best Avg MAE` 时晋级。
3. 其他情况不晋级，丢弃本轮新增模块。
4. 禁止根据单个 horizon 决定晋级。

初始化：

```text
best_name    = baseline
best_avg_mse = 0.4342
best_avg_mae = 0.4491
accepted_timestamp = false
accepted_reprogram = false
accepted_prompt    = false
```

## 阶段一：验证自回归核心

运行：

```bash
source /home/Lain/anaconda3/etc/profile.d/conda.sh
conda activate time-llm
bash scripts/TimeLLM_AutoTimes_ETTh1.sh ar_direct
```

找到本轮最新日志：

```bash
ls -1t logs/ETTh1/ETTh1_512_ar_direct_*.log | head -n 1
```

解析并与 baseline 比较：

- 若 `ar_direct` 晋级：更新当前 best，进入阶段二。
- 若训练成功但不晋级：停止全部后续实验，输出最终报告；不要尝试 timestamp、reprogram、Prompt或学习率调整。
- 若训练失败：按照“故障处理规则”修复并重新运行 `ar_direct`，不能将失败当成精度不佳。

## 阶段二：测试 GPT-2 textual timestamp

仅在 `ar_direct` 晋级后执行。

若缓存不存在，先生成：

```bash
python preprocess_gpt2_timestamps.py \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --output ./dataset/ETT-small/ETTh1_gpt2_tl64.pt \
  --token_len 64 \
  --batch_size 256 \
  --device cuda:0
```

确认缓存 metadata 中：

```text
data_path=ETTh1.csv
token_len=64
hidden_dim=768
model_id=AI-ModelScope/gpt2
num_rows=17420
```

运行：

```bash
bash scripts/TimeLLM_AutoTimes_ETTh1.sh ar_timestamp
```

与当前 best 比较：

- 晋级：`accepted_timestamp=true`，更新 best。
- 不晋级：`accepted_timestamp=false`，保持原 best。
- 无论是否晋级，只要实验成功完成，都进入阶段三。

## 阶段三：测试门控 TimeLLM reprogramming

根据 timestamp 是否被接受选择唯一命令：

```bash
# accepted_timestamp=true
bash scripts/TimeLLM_AutoTimes_ETTh1.sh ar_timestamp_reprogram

# accepted_timestamp=false
bash scripts/TimeLLM_AutoTimes_ETTh1.sh ar_reprogram
```

禁止两个命令都运行。

与当前 best 比较：

- 晋级：`accepted_reprogram=true`，更新 best。
- 不晋级：`accepted_reprogram=false`，保持原 best。
- 实验成功后进入阶段四。

## 阶段四：测试 TimeLLM 统计 Prompt

根据已经接受的模块选择唯一 variant：

| timestamp | reprogram | variant |
|---|---|---|
| false | false | `ar_prompt` |
| true | false | `ar_timestamp_prompt` |
| false | true | `ar_reprogram_prompt` |
| true | true | `ar_timestamp_reprogram_prompt` |

运行：

```bash
bash scripts/TimeLLM_AutoTimes_ETTh1.sh <variant>
```

与当前 best 比较：

- 晋级：`accepted_prompt=true`，更新 best。
- 不晋级：`accepted_prompt=false`，保持原 best。
- 完成后停止实验，不再增加模块或调参。

## 长训练监控规则

1. 启动前检查是否已有相同实验：

   ```bash
   ps -ef | rg 'run_autotimes.py|TimeLLM_AutoTimes_ETTh1.sh'
   nvidia-smi
   ```

2. 若同一 variant 正在运行，继续监控，绝不能重复启动。
3. 每次监控至少检查：进程仍存在、GPU显存/利用率、最新日志尾部、已完成 epoch 数、是否出现 traceback/OOM/NaN。
4. 单个 epoch 长时间没有新日志不代表卡死；只要进程存在且GPU仍在工作，就继续等待。
5. 不得因为验证或720滚动评估耗时较长而杀进程。
6. 只有满足以下任一条件才判定失败：
   - 进程退出且脚本返回非零状态；
   - 日志出现未处理 traceback；
   - CUDA OOM；
   - 任一 horizon 在全部 epoch 都没有 `status=ok` 指标；
   - 日志汇总脚本返回失败。
7. 正常完成必须满足以下任一条件，并且96/192/336/720各自至少有一条 `status=ok`：
   - 日志出现 `TRAINING_STOP reason=early_stopping`，且对应 `counter=3 patience=3 stop=true`；
   - 完整训练到第100个 epoch。

## 故障处理规则

- 先读取完整 traceback、相关代码和当前配置，确定根因后再修改。
- 只允许修复新加入的 AutoTimes 文件：

  ```text
  models/TimeLLM_AutoTimes.py
  data_provider/autotimes_data_loader.py
  run_autotimes.py
  preprocess_gpt2_timestamps.py
  summarize_autotimes_log.py
  scripts/TimeLLM_AutoTimes_ETTh1.sh
  tests/test_timellm_autotimes.py
  ```

- 不得通过缩小 batch、减少 epoch、减少 horizon、关闭模块、改变精度口径来绕过错误。
- 修复后运行语法检查、测试和单 batch 冒烟验证，再从失败阶段重新训练；因为没有 checkpoint，不能伪造续训。
- 对不确定的模型设计问题停止修改并询问 Lain；不得自行改变实验语义。

## 状态记录

维护 `autotimes_loop_status.md`，每个阶段结束后追加：

```text
variant
日志绝对路径
运行状态
96/192/336/720的best MSE、同epoch MAE和epoch
Avg MSE
Avg MAE
是否晋级
当前best及已接受模块
故障与修复记录（如有）
```

不要修改 `baseline.md`。不要提交或推送代码，除非 Lain 另行要求。

## 最终交付

完成或触发停止条件后，给 Lain 一份简洁报告：

1. 所有成功运行的 variant 与日志路径。
2. 每个 variant 的四 horizon 指标及平均值。
3. 每一步晋级或淘汰的计算依据。
4. 最终最佳模块组合。
5. 相对 baseline 的绝对与百分比 MSE/MAE变化。
6. 总训练耗时、速度、峰值显存。
7. 失败、修复及仍存在的限制。

最终报告必须基于日志，不得估计、补写或引用论文结果代替本地实验结果。
