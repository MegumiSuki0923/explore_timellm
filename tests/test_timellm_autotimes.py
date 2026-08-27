import torch

from models.TimeLLM_AutoTimes import build_shift_target, rollout_steps, segment_series
from summarize_autotimes_log import parse_best_metrics


def test_segment_series_covers_all_points_without_overlap():
    x = torch.arange(512).view(1, 512, 1)
    segments = segment_series(x, 64)
    assert segments.shape == (1, 8, 64)
    assert torch.equal(segments.reshape(1, 512), x[:, :, 0])


def test_shift_target_aligns_the_next_segment():
    x = torch.arange(512).view(1, 512, 1)
    future = torch.arange(512, 576).view(1, 64, 1)
    target = build_shift_target(x, future, 64)
    assert target.shape == x.shape
    assert torch.equal(target[:, :64], x[:, 64:128])
    assert torch.equal(target[:, -64:], future)


def test_rollout_steps_cover_and_truncate_all_horizons():
    assert [rollout_steps(horizon, 64) for horizon in (96, 192, 336, 720)] == [
        2,
        3,
        6,
        12,
    ]


def test_non_divisible_sequence_is_rejected():
    x = torch.zeros(1, 500, 1)
    try:
        segment_series(x, 64)
    except ValueError as error:
        assert "divisible" in str(error)
    else:
        raise AssertionError("expected a ValueError")


def test_log_parser_ignores_failed_and_nonfinite_metrics():
    lines = [
        "METRIC epoch=1 horizon=96 mse=0.4 mae=0.5 seconds=1 peak_mem_mb=1 status=ok",
        "METRIC epoch=2 horizon=96 mse=nan mae=0.1 seconds=1 peak_mem_mb=1 status=fail",
        "METRIC epoch=3 horizon=96 mse=0.3 mae=0.4 seconds=1 peak_mem_mb=1 status=ok",
    ]
    assert parse_best_metrics(lines) == {96: {"epoch": 3, "mse": 0.3, "mae": 0.4}}
