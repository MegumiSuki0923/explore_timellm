"""Smoke test for the ported LightGTS period-parallel decoding trio.

Run: python tests/smoke_period_decoder.py
"""
import sys
import types
from math import ceil

import torch

sys.path.insert(0, '.')

from layers.PeriodDecoder import Decoder, decoder_PredictHead

torch.manual_seed(0)

B, N, PATCH_NUMS, D_FF = 8, 7, 64, 128
PERIOD_LEN = 24


def trio(pred_len):
    out_patch_num = ceil(pred_len / PERIOD_LEN)
    decoder = Decoder(d_layers=3, patch_len=16, d_model=D_FF, n_heads=8,
                      d_ff=2 * D_FF, attn_dropout=0.4, dropout=0.)
    head = decoder_PredictHead(D_FF, 48, dropout=0.1)

    dec_cross = torch.randn(B, N, PATCH_NUMS, D_FF)
    # decoder_predict (LightGTS_pretrain_period.py:113-132)
    dec_in = dec_cross[:, :, -1, :].unsqueeze(2).expand(-1, -1, out_patch_num, -1)
    weights = 0.5 ** torch.arange(out_patch_num)
    dec_in = dec_in * weights.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    dec_out = decoder(dec_in, dec_cross)          # (B, N, out_patch_num, d_ff)
    dec_out = dec_out.transpose(2, 3)             # (B, N, d_ff, out_patch_num) — decoder_predict 的收尾转置
    y = head(dec_out, PERIOD_LEN)                 # (B, out_patch_num * period, N)
    y = y[:, :pred_len, :].permute(0, 2, 1).reshape(B * N, pred_len)

    assert y.shape == (B * N, pred_len), f"pred_len={pred_len}: got {y.shape}"
    assert torch.isfinite(y).all(), f"pred_len={pred_len}: non-finite values"
    n_params = sum(p.numel() for p in decoder.parameters()) + sum(p.numel() for p in head.parameters())
    return n_params


print("== PeriodDecoder trio smoke test ==")
for pred_len in [96, 192, 336, 720]:
    n = trio(pred_len)
    print(f"  pred_len={pred_len:>3}: output ok, trio params={n:,}")

# gradient flow check
pred_len = 96
decoder = Decoder(d_layers=3, patch_len=16, d_model=D_FF, n_heads=8,
                  d_ff=2 * D_FF, attn_dropout=0.4, dropout=0.)
head = decoder_PredictHead(D_FF, 48, dropout=0.1)
dec_cross = torch.randn(B, N, PATCH_NUMS, D_FF, requires_grad=True)
out_patch_num = ceil(pred_len / PERIOD_LEN)
dec_in = dec_cross[:, :, -1, :].unsqueeze(2).expand(-1, -1, out_patch_num, -1)
dec_in = dec_in * (0.5 ** torch.arange(out_patch_num)).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
y = head(decoder(dec_in, dec_cross).transpose(2, 3), PERIOD_LEN)
y.sum().backward()
dec_cross_grad = dec_cross.grad
assert dec_cross_grad is not None and torch.isfinite(dec_cross_grad).all()
n_grad = sum(1 for p in list(decoder.parameters()) + list(head.parameters()) if p.grad is not None)
n_total = len(list(decoder.parameters()) + list(head.parameters()))
print(f"  gradient flow: dec_cross grad ok, {n_grad}/{n_total} params received grads")
print("ALL PASSED")
