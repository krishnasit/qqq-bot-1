"""
Pattern / confluence detection for entries near the 50 EMA.
Compatibility version for the live paper bot.
"""
from __future__ import annotations

import pandas as pd


def _body(df):
    return (df["close"] - df["open"]).abs()


def _range(df):
    return (df["high"] - df["low"]).replace(0, 1e-9)


def bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    prev_bear = df["close"].shift(1) < df["open"].shift(1)
    curr_bull = df["close"] > df["open"]
    engulfs = (df["close"] >= df["open"].shift(1)) & (df["open"] <= df["close"].shift(1))
    return prev_bear & curr_bull & engulfs


def bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    prev_bull = df["close"].shift(1) > df["open"].shift(1)
    curr_bear = df["close"] < df["open"]
    engulfs = (df["open"] >= df["close"].shift(1)) & (df["close"] <= df["open"].shift(1))
    return prev_bull & curr_bear & engulfs


def bullish_pin_bar(df: pd.DataFrame, wick_body_ratio_min: float) -> pd.Series:
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]
    body = _body(df)
    return (lower_wick >= wick_body_ratio_min * body.replace(0, 1e-9)) & (df["close"] > df["open"])


def bearish_pin_bar(df: pd.DataFrame, wick_body_ratio_min: float) -> pd.Series:
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    body = _body(df)
    return (upper_wick >= wick_body_ratio_min * body.replace(0, 1e-9)) & (df["close"] < df["open"])


def inside_bar_break_bullish(df: pd.DataFrame) -> pd.Series:
    inside = (df["high"].shift(1) <= df["high"].shift(2)) & (df["low"].shift(1) >= df["low"].shift(2))
    breaks_up = df["close"] > df["high"].shift(1)
    return inside & breaks_up


def inside_bar_break_bearish(df: pd.DataFrame) -> pd.Series:
    inside = (df["high"].shift(1) <= df["high"].shift(2)) & (df["low"].shift(1) >= df["low"].shift(2))
    breaks_down = df["close"] < df["low"].shift(1)
    return inside & breaks_down


def wick_rejection_bullish(df: pd.DataFrame, wick_body_ratio_min: float) -> pd.Series:
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]
    body = _body(df)
    return lower_wick >= wick_body_ratio_min * body.replace(0, 1e-9)


def wick_rejection_bearish(df: pd.DataFrame, wick_body_ratio_min: float) -> pd.Series:
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    body = _body(df)
    return upper_wick >= wick_body_ratio_min * body.replace(0, 1e-9)


def liquidity_sweep_bullish(df: pd.DataFrame) -> pd.Series:
    swept = df["low"] < df["swing_low"].shift(1)
    reclaimed = df["close"] > df["swing_low"].shift(1)
    return swept & reclaimed


def liquidity_sweep_bearish(df: pd.DataFrame) -> pd.Series:
    swept = df["high"] > df["swing_high"].shift(1)
    reclaimed = df["close"] < df["swing_high"].shift(1)
    return swept & reclaimed


def near_support_resistance(df: pd.DataFrame, proximity_pct: float) -> pd.Series:
    near_swing_low = (df["close"] - df["swing_low"]).abs() / df["close"] * 100 <= proximity_pct
    near_swing_high = (df["close"] - df["swing_high"]).abs() / df["close"] * 100 <= proximity_pct
    return near_swing_low | near_swing_high


def volume_confirms(df: pd.DataFrame, multiplier_min: float) -> pd.Series:
    return df["volume"] >= multiplier_min * df["vol_avg"]


def detect_pattern(df: pd.DataFrame, i: int, bias: str, enabled_patterns) -> str | None:
    enabled = {str(p).lower() for p in (enabled_patterns or [])}
    idx = df.index[i]

    # 1. Structural Liquidity Sweep (PDL/PDH)
    # We check if the current candle swept the previous day's extreme and reclaimed it
    if "liquidity_sweep" in enabled:
        pdl = df["prev_day_low"].iloc[i]
        pdh = df["prev_day_high"].iloc[i]

        if bias == "long":
            # Swept PDL and closed above it
            if df["low"].iloc[i] < pdl and df["close"].iloc[i] > pdl:
                return "liq_sweep_pdl"
        if bias == "short":
            # Swept PDH and closed below it
            if df["high"].iloc[i] > pdh and df["close"].iloc[i] < pdh:
                return "liq_sweep_pdh"

    # 2. RSI Divergence (The Momentum Shift)
    if "rsi_div" in enabled:
        if bias == "long" and bool(detect_bullish_divergence(df, i)):
            return "rsi_bull_div"
        if bias == "short" and bool(detect_bearish_divergence(df, i)):
            return "rsi_bear_div"

    # 3. Candlestick Confirmation
    if "engulfing" in enabled:
        if bias == "long" and bool(bullish_engulfing(df).loc[idx]):
            return "engulfing"
        if bias == "short" and bool(bearish_engulfing(df).loc[idx]):
            return "engulfing"

    if "pin_bar" in enabled:
        if bias == "long" and bool(bullish_pin_bar(df, 2.0).loc[idx]):
            return "pin_bar"
        if bias == "short" and bool(bearish_pin_bar(df, 2.0).loc[idx]):
            return "pin_bar"

    return None

def detect_bullish_divergence(df: pd.DataFrame, i: int) -> bool:
    if i < 20: return False
    # Price: Lower Low, RSI: Higher Low
    price_low = df["low"].iloc[i]
    prev_low = df["low"].iloc[i-10:i].min()
    rsi_now = df["rsi"].iloc[i]
    rsi_prev = df["rsi"].iloc[i-10:i].min()
    return (price_low < prev_low) and (rsi_now > rsi_prev)

def detect_bearish_divergence(df: pd.DataFrame, i: int) -> bool:
    if i < 20: return False
    # Price: Higher High, RSI: Lower High
    price_high = df["high"].iloc[i]
    prev_high = df["high"].iloc[i-10:i].max()
    rsi_now = df["rsi"].iloc[i]
    rsi_prev = df["rsi"].iloc[i-10:i].max()
    return (price_high > prev_high) and (rsi_now < rsi_prev)

