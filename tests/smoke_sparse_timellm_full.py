"""Full-model forward smoke test for SparseTimeLLM with the LightGTS trio head.

Run: python tests/smoke_sparse_timellm_full.py
"""
import sys
from argparse import Namespace

import torch

sys.path.insert(0, '.')

from models.SparseTimeLLM import Model

torch.manual_seed(0)

configs = Namespace(
    task_name='long_term_forecast',
    features='M',
    seq_len=512,
    label_len=48,
    pred_len=96,
    enc_in=7, dec_in=7, c_out=7,
    d_model=32, d_ff=128, n_heads=8,
    llm_model='GPT2', llm_layers=12, llm_dim=768,
    llm_chunk_size=56,
    patch_len=16, stride=8,
    period_len=24,
    dropout=0.1,
    prompt_domain=0,
)

model = Model(configs)
model.eval()

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"trainable params total: {n_params:,}")

x_enc = torch.randn(4, 512, 7)
x_mark_enc = torch.zeros(4, 512, 4)
x_dec = torch.zeros(4, 48, 7)
x_mark_dec = torch.zeros(4, 48, 4)

with torch.no_grad():
    out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

assert out.shape == (4, 96, 7), f"got {out.shape}"
assert torch.isfinite(out).all(), "non-finite output"
print(f"forward ok, output shape: {tuple(out.shape)}")

# parameter accounting: old vs new head
head_params = sum(p.numel() for p in model.head.parameters()) + \
              sum(p.numel() for p in model.decoder.parameters())
map_params = sum(p.numel() for p in model.mapping_layer.parameters())
print(f"  decoder+head trio: {head_params:,}")
print(f"  mapping_layer (untouched): {map_params:,}")
print("ALL PASSED")
