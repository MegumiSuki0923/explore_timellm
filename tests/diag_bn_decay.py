"""诊断：BatchNorm 在 0.5^i 衰减输入下的行为（720 时 30 个块跨 9 个数量级）。

对比 BatchNorm（LightGTS 默认）与 LayerNorm 下，decoder 输出的"块间可区分性"。
"""
import sys
sys.path.insert(0, '.')
import torch

from layers.PeriodDecoder import DecoderLayer

torch.manual_seed(0)
B, N, PATCH, D = 8, 7, 64, 128
OUT = 30  # pred_len=720 -> out_patch_num=30

dec_cross = torch.randn(B, N, PATCH, D)
last = dec_cross[:, :, -1, :].unsqueeze(2).expand(-1, -1, OUT, -1)
weights = 0.5 ** torch.arange(OUT)
dec_in = last * weights.unsqueeze(0).unsqueeze(0).unsqueeze(-1)

print(f"dec_in 块间幅度: 0.5^0={weights[0]:.1f} ... 0.5^10={weights[10]:.2e} ... 0.5^29={weights[29]:.2e}")

for norm in ["BatchNorm", "LayerNorm"]:
    torch.manual_seed(0)
    layers = [DecoderLayer(16, D, 8, 2 * D, 0.4, 0., norm=norm) for _ in range(3)]
    def dec(x, cross):
        for l in layers:
            x = l(x, cross)
        return x
    for l in layers: l.eval()  # 用 running stats（模拟推理）
    with torch.no_grad():
        out = dec(dec_in, dec_cross)  # (B, N, 30, D)
    # 块间可区分性：相邻块输出的余弦相似度（越大越不可区分）
    v = out[0, 0]  # (30, D)
    cos = torch.nn.functional.cosine_similarity(v[:-1], v[1:], dim=-1)
    # 后 15 个块（本该反映"远端未来"差异的位置）
    late_cos = cos[14:].mean().item()
    early_cos = cos[:14].mean().item()
    print(f"[{norm:>9}] 相邻块余弦相似度: 前14块={early_cos:.4f}, 后15块={late_cos:.6f}  (→1.0 表示输出无法区分远端块)")
