import argparse
from pathlib import Path

import pandas as pd
import torch
from modelscope.hub.snapshot_download import snapshot_download
from torch.utils.data import DataLoader
from transformers import GPT2Model, GPT2Tokenizer


def infer_frequency(dates: pd.Series) -> pd.Timedelta:
    differences = dates.diff().dropna()
    if differences.empty:
        raise ValueError("at least two timestamps are required")
    modes = differences.mode()
    return modes.iloc[0] if not modes.empty else differences.median()


def build_timestamp_texts(
    dates: pd.Series, token_len: int, frequency: pd.Timedelta
) -> list[str]:
    duration = frequency * (token_len - 1)
    return [
        f"This is Time Series from {start} to {start + duration}"
        for start in dates
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute GPT-2 timestamp embeddings")
    parser.add_argument("--root_path", default="./dataset/ETT-small/")
    parser.add_argument("--data_path", default="ETTh1.csv")
    parser.add_argument("--output", default="./dataset/ETT-small/ETTh1_gpt2_tl64.pt")
    parser.add_argument("--token_len", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    csv_path = Path(args.root_path) / args.data_path
    frame = pd.read_csv(csv_path)
    dates = pd.to_datetime(frame["date"])
    frequency = infer_frequency(dates)
    texts = build_timestamp_texts(dates, args.token_len, frequency)

    local_model_path = snapshot_download("AI-ModelScope/gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained(local_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2Model.from_pretrained(local_model_path).to(args.device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

    embeddings = []
    loader = DataLoader(texts, batch_size=args.batch_size, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            encoded = tokenizer(
                list(batch),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )
            input_ids = encoded.input_ids.to(args.device)
            attention_mask = encoded.attention_mask.to(args.device)
            hidden = model(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state
            last_indices = attention_mask.sum(dim=1) - 1
            row_indices = torch.arange(hidden.shape[0], device=hidden.device)
            embeddings.append(hidden[row_indices, last_indices].float().cpu())

    payload = {
        "embeddings": torch.cat(embeddings, dim=0),
        "metadata": {
            "model_id": "AI-ModelScope/gpt2",
            "model_revision": Path(local_model_path).name,
            "hidden_dim": model.config.hidden_size,
            "token_len": args.token_len,
            "frequency": str(frequency),
            "data_path": Path(args.data_path).name,
            "num_rows": len(frame),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(f"saved {payload['embeddings'].shape} to {output}")


if __name__ == "__main__":
    main()
