# TimeLLM-GPT2 × AutoTimes 精度改造设计

## 目标与基线

在不修改原始 `TimeLLM.py`、`Embed.py` 和 `baseline.md` 的前提下，新增独立的 GPT-2-only 自回归模型。当前基线为 ETTh1、`seq_len=512`、Avg MSE `0.4342`、Avg MAE `0.4491`。

固定配置：`seq_len=512`、`label_len=448`、`token_len=64`、`batch_size=24`、`lr=5e-4`、seed `2021`、bf16、10 epochs、无 checkpoint，仅写日志。

## 核心架构

输入 `[B,512,C]` 按 Channel Independence 转成 `[B*C,8,64]`。两层 MLP `64→512→768` 生成 segment embedding，冻结 GPT-2 输出逐 token hidden state，两层 MLP `768→512→64` 解码下一 segment。

训练监督为整体右移一个 segment：`concat(x_enc[:,64:,:], future[:,:64,:])`，八个 token 全部计算 MSE。推理每次生成64点、窗口左移64点并回填，96/192/336/720分别滚动2/3/6/12次后精确截断。

## 独立增量模块

1. `ar_direct`：纯 AutoTimes 自回归核心，无 Prompt、时间戳和重编程。
2. `timestamp`：同一冻结 GPT-2 离线编码64点时间区间文本，取最后有效 token 的768维 hidden state，与数值 embedding 归一化后通过可学习标量相加。
3. `reprogram`：TimeLLM 的 TokenEmbedding、1000词表原型和 ReprogrammingLayer作为门控残差，`alpha=-4`，融合为 `LayerNorm(E_base + sigmoid(alpha)*LayerNorm(E_reprogram))`。
4. `prompt`：最后的独立消融；复用 TimeLLM 统计 Prompt，任务描述改成预测下一64点。

只有 `ar_direct` Avg MSE 低于0.4342才继续可选模块。可选模块逐一与当前最佳结构比较，变差即丢弃，但不阻止测试其他模块。

## 数据与时间戳

训练数据只需要未来64点；测试分别提供96/192/336/720真值。未来时间戳属于已知协变量，可按采样频率外推；任何未来数值不得进入输入或 Prompt。

时间戳缓存必须记录数据集、GPT-2 revision、频率、token length与hidden size；不匹配时明确失败。所有向量归一化加入 epsilon。

## 运行和指标

新增独立模型、独立 runner 和独立 ETTh1 脚本。每个 epoch 直接滚动评估四个 horizon，记录 Test MSE/MAE、速度、耗时和峰值显存。日志路径为 `./logs/ETTh1/ETTh1_512_{model_comment}_{YYYY-mm-dd_HH:mm}.log`。

每个 horizon 从日志选择最低 Test MSE epoch，并取同 epoch MAE；四项平均 MSE 为主指标，差值小于0.001时以 Avg MAE 决胜。第一轮不搜索学习率。

## 验收

- 512点严格形成8个segment，无补齐和裁剪。
- 训练输出 `[B,512,C]`，单步输出 `[B,64,C]`。
- 递增合成序列验证下一segment标签对齐。
- 四个 horizon 的循环次数和截断正确。
- GPT-2 全冻结；MLP与已启用增量模块可训练。
- 无 NaN/Inf、无未来数值泄漏。
- 原始基线三个文件实施前后哈希一致。
