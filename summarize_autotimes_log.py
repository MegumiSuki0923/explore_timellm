import argparse
import math
import re
from pathlib import Path


METRIC_PATTERN = re.compile(
    r"^METRIC epoch=(?P<epoch>\d+) horizon=(?P<horizon>\d+) "
    r"mse=(?P<mse>\S+) mae=(?P<mae>\S+) .* status=(?P<status>\w+)"
)


def parse_best_metrics(lines):
    best = {}
    for line in lines:
        match = METRIC_PATTERN.search(line.strip())
        if not match or match.group("status") != "ok":
            continue
        mse = float(match.group("mse"))
        mae = float(match.group("mae"))
        if not math.isfinite(mse) or not math.isfinite(mae):
            continue
        horizon = int(match.group("horizon"))
        record = {
            "epoch": int(match.group("epoch")),
            "mse": mse,
            "mae": mae,
        }
        if horizon not in best or mse < best[horizon]["mse"]:
            best[horizon] = record
    return best


def main():
    parser = argparse.ArgumentParser(description="Summarize TimeLLM_AutoTimes logs")
    parser.add_argument("log")
    args = parser.parse_args()
    best = parse_best_metrics(Path(args.log).read_text().splitlines())
    required = (96, 192, 336, 720)
    missing = [horizon for horizon in required if horizon not in best]
    if missing:
        raise SystemExit(f"missing successful metrics for horizons: {missing}")
    for horizon in required:
        record = best[horizon]
        print(
            f"horizon={horizon} mse={record['mse']:.4f} "
            f"mae={record['mae']:.4f} epoch={record['epoch']}"
        )
    print(f"avg_mse={sum(best[h]['mse'] for h in required)/4:.4f}")
    print(f"avg_mae={sum(best[h]['mae'] for h in required)/4:.4f}")


if __name__ == "__main__":
    main()
