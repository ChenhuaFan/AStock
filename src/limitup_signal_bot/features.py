from __future__ import annotations

import numpy as np


SEQ_FEATURE_NAMES = [
    "open_to_prev_close",
    "close_to_prev_close",
    "high_to_prev_close",
    "low_to_prev_close",
    "close_to_open",
    "range_to_prev_close",
    "upper_shadow_to_prev_close",
    "lower_shadow_to_prev_close",
    "amplitude_pct",
    "turnover_pct",
    "volume_log_ratio_5",
    "volume_log_ratio_20",
    "amount_log_ratio_5",
    "amount_log_ratio_20",
    "turnover_ratio_5",
    "turnover_ratio_20",
    "close_to_ma5",
    "close_to_ma10",
    "close_to_ma20",
    "position_20",
]


def safe_div(numerator: np.ndarray, denominator: np.ndarray | float) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float32),
        where=np.abs(denominator) > 1e-12,
    )


def trailing_mean_axis1(values: np.ndarray, period: int) -> np.ndarray:
    out = np.empty_like(values, dtype=np.float32)
    cumsum = np.cumsum(values.astype(np.float64), axis=1)
    for i in range(values.shape[1]):
        start = max(0, i - period + 1)
        total = cumsum[:, i] - (cumsum[:, start - 1] if start else 0.0)
        out[:, i] = total / (i - start + 1)
    return out


def trailing_min_axis1(values: np.ndarray, period: int) -> np.ndarray:
    out = np.empty_like(values, dtype=np.float32)
    for i in range(values.shape[1]):
        start = max(0, i - period + 1)
        out[:, i] = np.min(values[:, start : i + 1], axis=1)
    return out


def trailing_max_axis1(values: np.ndarray, period: int) -> np.ndarray:
    out = np.empty_like(values, dtype=np.float32)
    for i in range(values.shape[1]):
        start = max(0, i - period + 1)
        out[:, i] = np.max(values[:, start : i + 1], axis=1)
    return out


def log_ratio_to_mean_axis1(values: np.ndarray, period: int) -> np.ndarray:
    mean = trailing_mean_axis1(values, period)
    return np.log1p(np.maximum(values, 0.0)) - np.log1p(np.maximum(mean, 0.0))


def batch_to_seq_features(x_raw: np.ndarray) -> np.ndarray:
    open_ = x_raw[:, :, 0].astype(np.float32)
    close = x_raw[:, :, 1].astype(np.float32)
    high = x_raw[:, :, 2].astype(np.float32)
    low = x_raw[:, :, 3].astype(np.float32)
    volume = x_raw[:, :, 4].astype(np.float32)
    amount = x_raw[:, :, 5].astype(np.float32)
    amplitude = x_raw[:, :, 6].astype(np.float32) / 100.0
    pct_chg = x_raw[:, :, 7].astype(np.float32) / 100.0
    change = x_raw[:, :, 8].astype(np.float32)
    turnover = x_raw[:, :, 9].astype(np.float32) / 100.0

    prev_close = close - change
    prev_close = np.where(np.abs(prev_close) > 1e-12, prev_close, safe_div(close, 1.0 + pct_chg))

    ma5 = trailing_mean_axis1(close, 5)
    ma10 = trailing_mean_axis1(close, 10)
    ma20 = trailing_mean_axis1(close, 20)
    low20 = trailing_min_axis1(low, 20)
    high20 = trailing_max_axis1(high, 20)
    position20 = safe_div(close - low20, high20 - low20)

    seq = np.stack(
        [
            safe_div(open_ - prev_close, prev_close),
            safe_div(close - prev_close, prev_close),
            safe_div(high - prev_close, prev_close),
            safe_div(low - prev_close, prev_close),
            safe_div(close - open_, open_),
            safe_div(high - low, prev_close),
            safe_div(high - np.maximum(open_, close), prev_close),
            safe_div(np.minimum(open_, close) - low, prev_close),
            amplitude,
            turnover,
            log_ratio_to_mean_axis1(volume, 5),
            log_ratio_to_mean_axis1(volume, 20),
            log_ratio_to_mean_axis1(amount, 5),
            log_ratio_to_mean_axis1(amount, 20),
            safe_div(turnover, trailing_mean_axis1(turnover, 5)) - 1.0,
            safe_div(turnover, trailing_mean_axis1(turnover, 20)) - 1.0,
            safe_div(close, ma5) - 1.0,
            safe_div(close, ma10) - 1.0,
            safe_div(close, ma20) - 1.0,
            position20 - 0.5,
        ],
        axis=2,
    )
    return np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def batch_max_drawdown(close: np.ndarray) -> np.ndarray:
    peaks = np.maximum.accumulate(close, axis=1)
    return np.min(safe_div(close - peaks, peaks), axis=1)


def batch_price_position(close: np.ndarray, high: np.ndarray, low: np.ndarray) -> np.ndarray:
    high_max = np.max(high, axis=1)
    low_min = np.min(low, axis=1)
    return safe_div(close[:, -1] - low_min, high_max - low_min) - 0.5


def batch_to_tab_features(x_raw: np.ndarray, x_seq: np.ndarray) -> np.ndarray:
    window = x_raw.shape[1]
    periods = [3, 5, 10, 20]
    if window >= 60:
        periods.append(60)
    if window >= 120:
        periods.append(120)

    close = x_raw[:, :, 1].astype(np.float32)
    high = x_raw[:, :, 2].astype(np.float32)
    low = x_raw[:, :, 3].astype(np.float32)
    volume = x_raw[:, :, 4].astype(np.float32)
    amount = x_raw[:, :, 5].astype(np.float32)
    amplitude = x_raw[:, :, 6].astype(np.float32) / 100.0
    returns = x_raw[:, :, 7].astype(np.float32) / 100.0
    turnover = x_raw[:, :, 9].astype(np.float32) / 100.0

    columns = [x_seq[:, -1, :]]
    for period in periods:
        p = min(period, window)
        close_p = close[:, -p:]
        high_p = high[:, -p:]
        low_p = low[:, -p:]
        volume_p = volume[:, -p:]
        amount_p = amount[:, -p:]
        amplitude_p = amplitude[:, -p:]
        returns_p = returns[:, -p:]
        turnover_p = turnover[:, -p:]

        stats = np.column_stack(
            [
                safe_div(close_p[:, -1] - close_p[:, 0], close_p[:, 0]),
                np.std(returns_p, axis=1),
                np.mean(returns_p, axis=1),
                batch_max_drawdown(close_p),
                batch_price_position(close_p, high_p, low_p),
                np.log1p(np.maximum(volume_p[:, -1], 0.0))
                - np.log1p(np.maximum(np.mean(volume_p, axis=1), 0.0)),
                np.log1p(np.maximum(amount_p[:, -1], 0.0))
                - np.log1p(np.maximum(np.mean(amount_p, axis=1), 0.0)),
                np.mean(turnover_p, axis=1),
                np.mean(amplitude_p, axis=1),
                np.mean(returns_p > 0.0, axis=1),
                np.sum(returns_p >= 0.099, axis=1),
                np.sum(returns_p <= -0.099, axis=1),
            ]
        )
        columns.append(stats.astype(np.float32))

    p3 = min(3, window)
    p20 = min(20, window)
    ret3 = safe_div(close[:, -1] - close[:, -p3], close[:, -p3])
    ret20 = safe_div(close[:, -1] - close[:, -p20], close[:, -p20])
    vol5 = np.std(returns[:, -min(5, window) :], axis=1)
    vol20 = np.std(returns[:, -p20:], axis=1)
    volume5 = np.mean(volume[:, -min(5, window) :], axis=1)
    volume20 = np.mean(volume[:, -p20:], axis=1)
    turnover5 = np.mean(turnover[:, -min(5, window) :], axis=1)
    turnover20 = np.mean(turnover[:, -p20:], axis=1)
    columns.append(
        np.column_stack(
            [
                ret3 - ret20,
                vol5 - vol20,
                np.log1p(np.maximum(volume5, 0.0)) - np.log1p(np.maximum(volume20, 0.0)),
                turnover5 - turnover20,
            ]
        ).astype(np.float32)
    )
    return np.nan_to_num(np.concatenate(columns, axis=1), nan=0.0, posinf=0.0, neginf=0.0)
