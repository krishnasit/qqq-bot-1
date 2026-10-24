from __future__ import annotations

import pandas as pd
import numpy as np


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

    # 1. HTF Bias (Using a 1-hour approximation from 5m bars)
    # 1h Trend = Price > SMA(20) of 1h Close
    resampled_1h = out["close"].resample("h").last().dropna()
    sma_1h = resampled_1h.rolling(20, min_periods=20).mean()

    # Preserve missing warm-up values; do not silently turn them into short signals.
    bias_1h = pd.Series(index=resampled_1h.index, dtype="object")
    bias_1h.loc[resampled_1h > sma_1h] = "long"
    bias_1h.loc[resampled_1h < sma_1h] = "short"

    # Map the last completed, valid hourly bias onto each 5-minute bar.
    out["bias"] = bias_1h.reindex(out.index, method="ffill")
    out["htf_bias_raw"] = out["bias"]

    # 2. Daily Levels for Structural Sweeps
    # Get daily high/low and shift by 1 to get "Previous Day"
    daily = out["close"].resample('D').ohlc()
    out["prev_day_high"] = daily["high"].shift(1).reindex(out.index, method='ffill')
    out["prev_day_low"] = daily["low"].shift(1).reindex(out.index, method='ffill')

    # 3. Momentum Indicator (RSI)
    delta = out["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi"] = 100 - (100 / (1 + rs))

    # 4. Existing logic for EMA and ATR
    out["ema"] = out["close"].ewm(span=int(config.entry_ema_length), adjust=False).mean()
    out["trend_sma"] = out["close"].rolling(int(config.trend_sma_length), min_periods=int(config.trend_sma_length)).mean()

    prev_close = out["close"].shift(1)
    tr = pd.concat([
        (out["high"] - out["low"]).abs(),
        (out["high"] - prev_close).abs(),
        (out["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
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
