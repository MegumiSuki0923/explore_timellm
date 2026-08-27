# TimeLLM-GPT2 × AutoTimes 精度改造设计

## 目标与基线

在不修改原始 `TimeLLM.py`、`Embed.py` 和 `baseline.md` 的前提下，新增独立的 GPT-2-only 自回归模型。当前基线为 ETTh1、`seq_len=512`、Avg MSE `0.4342`、Avg MAE `0.4491`。

固定配置：`seq_len=512`、`label_len=448`、`token_len=64`、`batch_size=24`、`lr=5e-4`、seed `2021`、bf16、10 epochs、无 checkpoint，仅写日志。

## 核心架构

复用当前已做 Channel Independence 的 loader：模型实际输入固定为单通道 `[B,512,1]`，重排成 `[B,8,64]`。两层 MLP `64→512→768` 生成 segment embedding，冻结 GPT-2 输出逐 token hidden state，两层 MLP `768→512→64` 解码下一 segment。

训练/验证 loader 固定 `pred_len=64`，`future=batch_y[:,-64:,:]`。训练监督为整体右移一个 segment：`concat(x_enc[:,64:,:], future)`，八个 token 全部计算 MSE。测试为96/192/336/720分别建立独立 loader。推理每次生成64点、窗口左移64点并回填，分别滚动2/3/6/12次后精确截断。

保留数据集 StandardScaler。模型每一步仅用当前512点上下文的 mean/std 归一化输入和下一段标签，并在归一化空间计算 loss；生成后先反归一化回数据尺度，再回填滑窗，下一步重新计算新窗口的 mean/std。

## 独立增量模块

1. `ar_direct`：纯 AutoTimes 自回归核心，无 Prompt、时间戳和重编程。
2. `timestamp`：同一冻结 GPT-2 离线编码64点时间区间文本，取最后有效 token 的768维 hidden state，与数值 embedding 归一化后通过可学习标量相加。
3. `reprogram`：TimeLLM 的 TokenEmbedding、1000词表原型和 ReprogrammingLayer作为门控残差，`alpha=-4`，融合为 `LayerNorm(E_base + sigmoid(alpha)*LayerNorm(E_reprogram))`。
4. `prompt`：最后的独立消融；复用 TimeLLM 统计 Prompt，任务描述改成预测下一64点。

只有 `ar_direct` Avg MSE 低于0.4342才继续可选模块。之后固定按 `timestamp → reprogram → prompt` 贪心累积：模块优于当前 best 就保留并作为下一模块基座，否则丢弃；不回溯、不测试其他排列。

## 数据与时间戳

训练数据只需要未来64点；测试四个 loader 分别提供96/192/336/720真值。未来时间戳属于已知协变量，可按采样频率外推；任何未来数值不得进入输入或 Prompt。

时间戳输入始终为8个与当前8个数值 segment 一一对应的64点区间 embedding。每滚动一步删除最旧1个区间、追加下一个未来64点区间，绝不按单时间点推进。

时间戳缓存必须记录数据集、GPT-2 revision、频率、token length与hidden size；不匹配时明确失败。所有向量归一化加入 epsilon。

## 运行和指标

新增独立模型、独立 runner 和独立 ETTh1 脚本。runner 禁用 early stopping、跑满10个 epoch、从不保存 checkpoint。每个 epoch 直接滚动评估四个 horizon，日志路径为 `./logs/ETTh1/ETTh1_512_{model_comment}_{YYYY-mm-dd_HH:mm}.log`。

每个 epoch/horizon 输出一行机器可解析记录：`METRIC epoch=<n> horizon=<h> mse=<v> mae=<v> seconds=<v> peak_mem_mb=<v> status=<ok|fail>`。NaN、Inf或异常记为 `status=fail`，该 epoch 不参与对应 horizon 的最优选择。

每个 horizon 从日志选择最低 Test MSE epoch，并取同 epoch MAE；四项平均 MSE 为主指标，差值小于0.001时以 Avg MAE 决胜。第一轮不搜索学习率。

## 验收

- 512点严格形成8个segment，无补齐和裁剪。
- loader 与模型固定单通道输入；训练输出 `[B,512,1]`，单步输出 `[B,64,1]`。
- 递增合成序列验证下一segment标签对齐。
- 四个 horizon 的循环次数和截断正确。
- 每次滚动按一个64点区间同步推进数值窗口和时间戳窗口。
- GPT-2 全冻结；MLP与已启用增量模块可训练。
- 无 NaN/Inf、无未来数值泄漏。
- 日志解析器能忽略 `status=fail` 并提取各 horizon 最低MSE及同epoch MAE。
- 原始基线三个文件实施前后哈希一致。
