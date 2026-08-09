from __future__ import annotations

import pandas as pd


def near_ema(row, ema_confluence_pct: float) -> bool:
    ema = row.get("ema")
    close = row.get("close")
    if pd.isna(ema) or pd.isna(close) or float(close) == 0:
        return False
    distance_pct = abs(float(close) - float(ema)) / float(close) * 100.0
    return distance_pct <= float(ema_confluence_pct)


def _compute_bias(df: pd.DataFrame, sma_length: int) -> pd.Series:
    sma = df["close"].rolling(sma_length, min_periods=sma_length).mean()
    bias = pd.Series(index=df.index, dtype="object")
    bias[df["close"] > sma] = "long"
    bias[df["close"] < sma] = "short"
    return bias


def _compute_after_buffer(index: pd.DatetimeIndex, london_timezone: str, start_hour: int, start_minute: int, no_trade_buffer_minutes: int) -> pd.Series:
    local_times = index.tz_convert(london_timezone)
    start_minutes = start_hour * 60 + start_minute + no_trade_buffer_minutes
    mins = local_times.hour * 60 + local_times.minute
    return pd.Series(mins >= start_minutes, index=index)


def prepare_strategy_frame(df: pd.DataFrame, config) -> pd.DataFrame:
    out = df.copy().sort_index()

    out["ema"] = out["close"].ewm(span=int(config.entry_ema_length), adjust=False).mean()
    out["trend_sma"] = out["close"].rolling(int(config.trend_sma_length), min_periods=int(config.trend_sma_length)).mean()
    out["bias"] = _compute_bias(out, int(config.trend_sma_length))

    prev_close = out["close"].shift(1)
    tr = pd.concat(
        [
            (out["high"] - out["low"]).abs(),
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr"] = tr.rolling(int(config.atr_length), min_periods=int(config.atr_length)).mean()

    out["swing_low"] = out["low"].rolling(10, min_periods=10).min()
    out["swing_high"] = out["high"].rolling(10, min_periods=10).max()
    out["vol_avg"] = out["volume"].rolling(20, min_periods=1).mean()

    out["after_buffer"] = _compute_after_buffer(
        out.index,
        config.london_timezone,
        int(config.bias_start_hour),
        int(config.bias_start_minute),
        int(config.no_trade_buffer_minutes),
    )

    return out
