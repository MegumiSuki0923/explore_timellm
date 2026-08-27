# 基线
- 日期：2026-08-27
- 状态：完成
- 训练日志：explore_timellm-explore/logs/ETTh1/ETTh1_512_base_2026-08-27_14:13.log
- 训练脚本：scripts/ETTh1.sh
- 实验方案：Time-LLM GPT2 原始代码，未做任何更改

- 参考来源：无
- 运行方式：bash scripts/ETTh1.sh
- 实验效果:

| pred_len | best MSE | best MAE(with best MSE) | best epoch | speed      |
| -------- | -------- | ----------------------- | ---------- | ---------- |
| 96       | 0.3840   | 0.4124                  | 5          | 13.19 it/s |
| 192      | 0.4296   | 0.4416                  | 7          | 13.18 it/s |
| 336      | 0.4730   | 0.4723                  | 3          | 13.06 it/s |
| 720      | 0.4501   | 0.4703                  | 4          | 12.96 it/s |
| Avg      | 0.4342   | 0.4491                  | -          | 13.10 it/s |
