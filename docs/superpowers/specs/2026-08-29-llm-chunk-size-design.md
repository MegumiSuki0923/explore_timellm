# SparseTimeLLM LLM Chunk Size 配置设计

## 目标

将 SparseTimeLLM 中写死的 LLM micro-chunk 大小改为命令行参数。默认保持 32，ETTh1 训练显式使用 56，使一个 `batch_size=8`、`enc_in=7` 的批次可以一次送入 GPT-2。

## 接口与数据流

- `run_main.py` 新增整数参数 `--llm_chunk_size`，默认值为 32。
- `SparseTimeLLM.Model` 在初始化时保存该参数，前向传播使用它切分 `B * N` 维度。
- `scripts/SparseTimeLLM_ETTh1.sh` 定义 `llm_chunk_size=56`，在四个预测长度实验中都显式传入该值，并在启动摘要中记录它。
- 不改变 TimeLLM、SparseTSF 分支、损失函数、batch size 或预测张量形状。

## 约束与验证

- 参数必须为正整数；命令行类型检查负责拒绝非整数，模型初始化负责拒绝小于 1 的值。
- 当 `B * N <= llm_chunk_size` 时只执行一次 LLM forward；大于该值时维持现有分块与拼接行为。
- 验证 Python 和 shell 语法、参数解析，以及脚本四处调用均传入 56。
