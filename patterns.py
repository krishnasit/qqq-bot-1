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

    if "inside_bar" in enabled:
        if bias == "long" and bool(inside_bar_break_bullish(df).loc[idx]):
            return "inside_bar"
        if bias == "short" and bool(inside_bar_break_bearish(df).loc[idx]):
            return "inside_bar"

    if "wick_rejection" in enabled:
        if bias == "long" and bool(wick_rejection_bullish(df, 2.0).loc[idx]):
            return "wick_rejection"
        if bias == "short" and bool(wick_rejection_bearish(df, 2.0).loc[idx]):
            return "wick_rejection"

    if "liquidity_sweep" in enabled:
        if bias == "long" and bool(liquidity_sweep_bullish(df).loc[idx]):
            return "liquidity_sweep"
        if bias == "short" and bool(liquidity_sweep_bearish(df).loc[idx]):
            return "liquidity_sweep"

    return None
