import argparse
import random
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

from data_provider.autotimes_data_loader import autotimes_data_provider
from models import TimeLLM_AutoTimes
from models.TimeLLM_AutoTimes import build_shift_target, rollout_steps


HORIZONS = (96, 192, 336, 720)


def parse_args():
    parser = argparse.ArgumentParser(description="TimeLLM-GPT2 x AutoTimes")
    parser.add_argument("--data", default="ETTh1")
    parser.add_argument("--root_path", default="./dataset/ETT-small/")
    parser.add_argument("--data_path", default="ETTh1.csv")
    parser.add_argument("--features", default="M")
    parser.add_argument("--target", default="OT")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--label_len", type=int, default=448)
    parser.add_argument("--token_len", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--train_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--llm_model", default="GPT2")
    parser.add_argument("--llm_layers", type=int, default=12)
    parser.add_argument("--llm_dim", type=int, default=768)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--d_ff", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--mlp_hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use_timestamp", action="store_true")
    parser.add_argument("--timestamp_cache", default="")
    parser.add_argument("--use_reprogram", action="store_true")
    parser.add_argument("--use_prompt", action="store_true")
    parser.add_argument("--model_comment", default="ar_direct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mixed_precision", choices=("bf16", "none"), default="bf16")
    args = parser.parse_args()
    if args.label_len != args.seq_len - args.token_len:
        raise ValueError("label_len must equal seq_len - token_len")
    if args.use_timestamp and not args.timestamp_cache:
        raise ValueError("--timestamp_cache is required with --use_timestamp")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def autocast_context(args):
    if args.mixed_precision == "bf16" and torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def metric_line(epoch, horizon, mse, mae, seconds, peak_mem, status):
    print(
        f"METRIC epoch={epoch} horizon={horizon} mse={mse:.7f} mae={mae:.7f} "
        f"seconds={seconds:.3f} peak_mem_mb={peak_mem:.1f} status={status}",
        flush=True,
    )


@torch.no_grad()
def evaluate_horizon(model, loader, horizon, args):
    model.eval()
    squared_error = 0.0
    absolute_error = 0.0
    element_count = 0
    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for batch_x, batch_y, x_time, y_time in loader:
        window = batch_x.float().to(args.device)
        true = batch_y[:, -horizon:, :].float().to(args.device)
        x_time = x_time.long().to(args.device)
        y_time = y_time.long().to(args.device)
        predictions = []
        steps = rollout_steps(horizon, args.token_len)
        for step in range(steps):
            with autocast_context(args):
                predicted = model.predict_next(
                    window, x_time if args.use_timestamp else None
                )
            predictions.append(predicted)
            window = torch.cat([window[:, args.token_len :, :], predicted], dim=1)
            x_time = torch.cat([x_time[:, 1:], y_time[:, step : step + 1]], dim=1)
        forecast = torch.cat(predictions, dim=1)[:, :horizon, :]
        difference = forecast - true
        squared_error += difference.square().sum().item()
        absolute_error += difference.abs().sum().item()
        element_count += difference.numel()

    seconds = time.time() - start_time
    peak_mem = (
        torch.cuda.max_memory_allocated() / (1024**2)
        if torch.cuda.is_available()
        else 0.0
    )
    return squared_error / element_count, absolute_error / element_count, seconds, peak_mem


@torch.no_grad()
def validate(model, loader, args):
    model.eval()
    loss_sum = 0.0
    count = 0
    for batch_x, batch_y, x_time, _ in loader:
        batch_x = batch_x.float().to(args.device)
        future = batch_y[:, -args.token_len :, :].float().to(args.device)
        x_time = x_time.long().to(args.device)
        with autocast_context(args):
            prediction, means, stdev = model(
                batch_x, x_time if args.use_timestamp else None
            )
            target = build_shift_target(batch_x, future, args.token_len)
            normalized_target = (target - means) / stdev
            loss = F.mse_loss(prediction, normalized_target)
        loss_sum += loss.item()
        count += 1
    return loss_sum / count


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the configured device")

    print(f"CONFIG {vars(args)}", flush=True)
    _, train_loader = autotimes_data_provider(args, "train", args.token_len)
    _, val_loader = autotimes_data_provider(args, "val", args.token_len)
    test_loaders = {
        horizon: autotimes_data_provider(args, "test", horizon)[1]
        for horizon in HORIZONS
    }

    model = TimeLLM_AutoTimes.Model(args).to(args.device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.train_epochs, eta_min=1e-8
    )
    print(
        f"PARAMS trainable={sum(p.numel() for p in trainable)} "
        f"total={sum(p.numel() for p in model.parameters())}",
        flush=True,
    )

    for epoch in range(1, args.train_epochs + 1):
        model.train()
        epoch_start = time.time()
        train_loss_sum = 0.0
        train_count = 0
        for batch_x, batch_y, x_time, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            batch_x = batch_x.float().to(args.device)
            future = batch_y[:, -args.token_len :, :].float().to(args.device)
            x_time = x_time.long().to(args.device)
            with autocast_context(args):
                prediction, means, stdev = model(
                    batch_x, x_time if args.use_timestamp else None
                )
                target = build_shift_target(batch_x, future, args.token_len)
                normalized_target = (target - means) / stdev
                loss = F.mse_loss(prediction, normalized_target)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            train_count += 1
        scheduler.step()
        validation_loss = validate(model, val_loader, args)
        print(
            f"EPOCH epoch={epoch} train_loss={train_loss_sum/train_count:.7f} "
            f"val_loss={validation_loss:.7f} seconds={time.time()-epoch_start:.3f} "
            f"lr={optimizer.param_groups[0]['lr']:.8f}",
            flush=True,
        )

        for horizon, loader in test_loaders.items():
            try:
                mse, mae, seconds, peak_mem = evaluate_horizon(
                    model, loader, horizon, args
                )
                if not np.isfinite(mse) or not np.isfinite(mae):
                    metric_line(epoch, horizon, mse, mae, seconds, peak_mem, "fail")
                else:
                    metric_line(epoch, horizon, mse, mae, seconds, peak_mem, "ok")
            except Exception as error:
                print(
                    f"METRIC epoch={epoch} horizon={horizon} mse=nan mae=nan "
                    f"seconds=0.000 peak_mem_mb=0.0 status=fail error={type(error).__name__}:{error}",
                    flush=True,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
