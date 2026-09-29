from __future__ import annotations
"""
ULTIMATE MAX SWEEP (checkpointed)
===================================
Multi-strategy, multi-parameter, walk-forward validated backtest sweep with
realistic execution costs (volatility-scaled slippage, commission, N-bar fill
delay). Four independent strategy families, each with its own parameter grid
and entry/exit logic, all sharing one trade-simulation engine:

  1. ema_confluence   - EMA/trend/pattern strategy (reuses backtester.py + patterns.py)
  2. breakout         - Donchian channel breakout
  3. mean_reversion   - RSI + Bollinger Band mean reversion
  4. trend_macd       - MACD crossover filtered by ADX trend strength

For every parameter combo we run:
  - In-sample backtest on the first WF_TRAIN_PCT of bars (train)
  - Out-of-sample backtest on the remaining bars (test)
Combos profitable on BOTH splits are flagged robust=True (filters overfitting).

CHECKPOINTING: every CHECKPOINT_EVERY completed combos, partial results are
written to ultimate_sweep_results.csv (overwritten, cumulative) and a live
stats snapshot (best PF so far, robust count so far) is printed, then the run
continues automatically until all combos are done.

Usage:
  python ultimate_sweep.py --symbol QQQ --days 720 --workers 4 --sample 1.0
"""
import argparse, itertools, os, random, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from alpaca.data.enums import DataFeed
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from backtester import prepare_strategy_frame, near_ema
from config import StrategyConfig
from patterns import detect_pattern

load_dotenv()

# ---- Realistic cost model ----
BASE_SLIPPAGE_PCT     = 0.0004
VOL_SLIPPAGE_MULT     = 0.15
COMMISSION_PER_SHARE  = 0.0002
FILL_DELAY_BARS       = 1
RISK_PCT              = 0.015
MAX_CAPITAL_PCT       = 0.25
MIN_RISK_PER_SHARE    = 0.50
MIN_TRADES_REQUIRED   = 10
WF_TRAIN_PCT          = 0.70
CHECKPOINT_EVERY      = 500


def fetch_bars(symbol: str, days: int) -> pd.DataFrame:
    api_key    = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not api_secret:
        raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_SECRET_KEY in .env")
    client = StockHistoricalDataClient(api_key, api_secret)
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        start=start, end=end,
        feed=DataFeed.IEX,
    )
    bars = client.get_stock_bars(req).df
    if bars.empty:
        return bars
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(symbol, level=0)
    bars.index = pd.to_datetime(bars.index, utc=True)
    return bars[["open", "high", "low", "close", "volume"]].sort_index()


def add_common_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    prev_close = out["close"].shift(1)
    tr = pd.concat([
        (out["high"] - out["low"]).abs(),
        (out["high"] - prev_close).abs(),
        (out["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    out["atr14"] = tr.rolling(14, min_periods=14).mean()
    return out


def adaptive_slippage_pct(row) -> float:
    atr = row["atr14"] if "atr14" in row.index else row.get("atr")
    close = row.get("close")
    if atr is None or close in (None, 0) or pd.isna(atr) or pd.isna(close):
        return BASE_SLIPPAGE_PCT
    vol_ratio = float(atr) / float(close)
    return BASE_SLIPPAGE_PCT + VOL_SLIPPAGE_MULT * vol_ratio


def simulate_trades(sf: pd.DataFrame, signals: pd.Series, side_map: pd.Series,
                     stop_fn, target_fn, capital: float):
    equity = capital
    peak = capital
    max_dd = 0.0
    wins = losses = 0
    gross_win = gross_loss = 0.0
    total_r = 0.0
    in_position = False
    side = entry = stop = target = qty = None
    n = len(sf)

    sig_idx = np.where(signals.values)[0]
    sig_ptr = 0

    for i in range(n):
        row = sf.iloc[i]
        if in_position:
            h, l = float(row["high"]), float(row["low"])
            hit_stop   = (side == "long"  and l <= stop)  or (side == "short" and h >= stop)
            hit_target = (side == "long"  and h >= target) or (side == "short" and l <= target)
            if hit_stop or hit_target:
                raw_exit = stop if hit_stop else target
                slip = adaptive_slippage_pct(row)
                exit_price = raw_exit * (1 - slip) if side == "long" else raw_exit * (1 + slip)
                risk_amt = abs(entry - stop)
                r_mult = -1.0 if hit_stop else (abs(target - entry) / risk_amt if risk_amt > 0 else 0.0)
                pnl = qty * risk_amt * r_mult
                pnl -= qty * COMMISSION_PER_SHARE * 2
                equity += pnl
                gross_win += max(pnl, 0)
                gross_loss += abs(min(pnl, 0))
                total_r += r_mult
                if pnl > 0: wins += 1
                else: losses += 1
                peak = max(peak, equity)
                dd = (peak - equity) / peak * 100 if peak > 0 else 0
                max_dd = max(max_dd, dd)
                in_position = False
            continue

        while sig_ptr < len(sig_idx) and sig_idx[sig_ptr] < i - FILL_DELAY_BARS:
            sig_ptr += 1
        if sig_ptr >= len(sig_idx):
            continue
        sig_i = sig_idx[sig_ptr]
        fill_i = sig_i + FILL_DELAY_BARS
        if fill_i != i:
            continue
        sig_ptr += 1

        this_side = side_map.iloc[sig_i]
        if this_side not in ("long", "short"):
            continue

        fill_row = sf.iloc[i]
        slip = adaptive_slippage_pct(fill_row)
        raw_entry = float(fill_row["open"])
        entry_price = raw_entry * (1 + slip) if this_side == "long" else raw_entry * (1 - slip)

        stop_price = stop_fn(sf.iloc[sig_i], this_side)
        if stop_price is None or pd.isna(stop_price):
            continue
        target_price = target_fn(entry_price, stop_price, this_side)
        risk = abs(entry_price - stop_price)
        if risk <= 0 or stop_price <= 0 or target_price <= 0:
            continue

        safe_risk = max(risk, MIN_RISK_PER_SHARE)
        qty_risk = int((equity * RISK_PCT) // safe_risk)
        qty_cap = int((equity * MAX_CAPITAL_PCT) // entry_price)
        q = max(1, min(qty_risk, qty_cap))

        in_position = True
        side = this_side
        entry = entry_price
        stop = stop_price
        target = target_price
        qty = q

    total_trades = wins + losses
    if total_trades < MIN_TRADES_REQUIRED:
        return None
    pf = gross_win / gross_loss if gross_loss > 0 else 999.0
    return {
        "total_trades": total_trades,
        "win_rate_pct": round(wins / total_trades * 100, 2),
        "profit_factor": round(pf, 3),
        "avg_r": round(total_r / total_trades, 3),
        "max_dd_pct": round(max_dd, 2),
        "net_pnl": round(equity - capital, 2),
        "return_pct": round((equity - capital) / capital * 100, 2),
        "final_equity": round(equity, 2),
    }


# ==== Strategy 1: EMA confluence + candlestick patterns ====
EMA_GRID = {
    "entry_ema_length":     [10, 20, 50, 100],
    "trend_sma_length":     [50, 100, 200],
    "ema_confluence_pct":   [0.15, 0.2, 0.5, 0.8, 1.2],
    "atr_length":           [14, 20],
    "atr_stop_buffer_mult": [0.1, 0.25, 0.4],
    "rr_target":            [1.5, 2.0, 2.5],
    "no_trade_buffer_minutes": [60, 90],
    "enabled_patterns": [
        ["engulfing", "pin_bar"],
        ["engulfing", "pin_bar", "liquidity_sweep"],
    ],
}

def run_ema_confluence(df: pd.DataFrame, params: dict, capital: float):
    cfg = StrategyConfig()
    for k, v in params.items():
        setattr(cfg, k, v)
    cfg.initial_capital = capital
    sf = prepare_strategy_frame(df, cfg)
    signals = pd.Series(False, index=sf.index)
    sides = pd.Series(None, index=sf.index, dtype="object")
    warmup = max(cfg.trend_sma_length, cfg.atr_length, cfg.entry_ema_length, 15)
    for i in range(warmup, len(sf) - 1):
        row = sf.iloc[i]
        if not bool(row.get("after_buffer", False)): continue
        bias = row.get("bias")
        if bias not in ("long", "short"): continue
        if not near_ema(row, cfg.ema_confluence_pct): continue
        if pd.isna(row.get("atr")): continue
        if not detect_pattern(sf, i, bias, cfg.enabled_patterns): continue
        signals.iloc[i] = True
        sides.iloc[i] = bias

    def stop_fn(row, side):
        buf = max(float(row["atr"]) * cfg.atr_stop_buffer_mult, max(0.05, float(row["close"]) * 0.0005))
        return float(row["low"]) - buf if side == "long" else float(row["high"]) + buf

    def target_fn(entry, stop, side):
        r = abs(entry - stop)
        return entry + r * cfg.rr_target if side == "long" else entry - r * cfg.rr_target

    return simulate_trades(sf, signals, sides, stop_fn, target_fn, capital)


# ==== Strategy 2: Donchian channel breakout ====
BREAKOUT_GRID = {
    "channel_len":  [10, 20, 30, 50],
    "atr_len":      [14, 20],
    "stop_atr_mult": [1.0, 1.5, 2.0],
    "rr_target":    [1.5, 2.0, 3.0],
}

def run_breakout(df: pd.DataFrame, params: dict, capital: float):
    sf = add_common_indicators(df)
    n = params["channel_len"]
    sf["hh"] = sf["high"].rolling(n, min_periods=n).max().shift(1)
    sf["ll"] = sf["low"].rolling(n, min_periods=n).min().shift(1)
    prev_close = sf["close"].shift(1)
    tr = pd.concat([
        (sf["high"] - sf["low"]).abs(),
        (sf["high"] - prev_close).abs(),
        (sf["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    sf["atr"] = tr.rolling(params["atr_len"], min_periods=params["atr_len"]).mean()

    long_sig = sf["close"] > sf["hh"]
    short_sig = sf["close"] < sf["ll"]
    signals = (long_sig | short_sig)
    signals.iloc[: max(n, params["atr_len"]) + 1] = False
    sides = pd.Series(None, index=sf.index, dtype="object")
    sides[long_sig] = "long"
    sides[short_sig] = "short"

    def stop_fn(row, side):
        if pd.isna(row.get("atr")): return None
        buf = float(row["atr"]) * params["stop_atr_mult"]
        return float(row["close"]) - buf if side == "long" else float(row["close"]) + buf

    def target_fn(entry, stop, side):
        r = abs(entry - stop)
        return entry + r * params["rr_target"] if side == "long" else entry - r * params["rr_target"]

    return simulate_trades(sf, signals, sides, stop_fn, target_fn, capital)


# ==== Strategy 3: RSI + Bollinger Band mean reversion ====
MEANREV_GRID = {
    "rsi_len":     [7, 14, 21],
    "rsi_oversold": [20, 25, 30],
    "rsi_overbought": [70, 75, 80],
    "bb_len":      [20, 30],
    "bb_std":      [2.0, 2.5],
    "rr_target":   [1.0, 1.5, 2.0],
}

def _rsi(close: pd.Series, length: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(length, min_periods=length).mean()
    loss = (-delta.clip(upper=0)).rolling(length, min_periods=length).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def run_mean_reversion(df: pd.DataFrame, params: dict, capital: float):
    sf = add_common_indicators(df)
    sf["rsi"] = _rsi(sf["close"], params["rsi_len"])
    ma = sf["close"].rolling(params["bb_len"], min_periods=params["bb_len"]).mean()
    std = sf["close"].rolling(params["bb_len"], min_periods=params["bb_len"]).std()
    sf["bb_lower"] = ma - params["bb_std"] * std
    sf["bb_upper"] = ma + params["bb_std"] * std

    long_sig = (sf["rsi"] < params["rsi_oversold"]) & (sf["close"] < sf["bb_lower"])
    short_sig = (sf["rsi"] > params["rsi_overbought"]) & (sf["close"] > sf["bb_upper"])
    signals = long_sig | short_sig
    sides = pd.Series(None, index=sf.index, dtype="object")
    sides[long_sig] = "long"
    sides[short_sig] = "short"

    def stop_fn(row, side):
        if pd.isna(row.get("atr14")): return None
        buf = float(row["atr14"]) * 1.0
        return float(row["close"]) - buf if side == "long" else float(row["close"]) + buf

    def target_fn(entry, stop, side):
        r = abs(entry - stop)
        return entry + r * params["rr_target"] if side == "long" else entry - r * params["rr_target"]

    return simulate_trades(sf, signals, sides, stop_fn, target_fn, capital)


# ==== Strategy 4: MACD crossover + ADX trend filter ====
TRENDMACD_GRID = {
    "fast":        [8, 12],
    "slow":        [21, 26],
    "signal":      [9],
    "adx_len":     [14],
    "adx_min":     [15, 20, 25],
    "rr_target":   [1.5, 2.0, 2.5, 3.0],
}

def _adx(df: pd.DataFrame, length: int) -> pd.Series:
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(length, min_periods=length).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(length, min_periods=length).mean() / atr.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(length, min_periods=length).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(length, min_periods=length).mean()

def run_trend_macd(df: pd.DataFrame, params: dict, capital: float):
    sf = add_common_indicators(df)
    ema_fast = sf["close"].ewm(span=params["fast"], adjust=False).mean()
    ema_slow = sf["close"].ewm(span=params["slow"], adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=params["signal"], adjust=False).mean()
    sf["adx"] = _adx(sf, params["adx_len"])

    cross_up = (macd.shift(1) < macd_signal.shift(1)) & (macd > macd_signal)
    cross_down = (macd.shift(1) > macd_signal.shift(1)) & (macd < macd_signal)
    trend_ok = sf["adx"] > params["adx_min"]

    long_sig = cross_up & trend_ok
    short_sig = cross_down & trend_ok
    signals = long_sig | short_sig
    sides = pd.Series(None, index=sf.index, dtype="object")
    sides[long_sig] = "long"
    sides[short_sig] = "short"

    def stop_fn(row, side):
        if pd.isna(row.get("atr14")): return None
        buf = float(row["atr14"]) * 1.5
        return float(row["close"]) - buf if side == "long" else float(row["close"]) + buf

    def target_fn(entry, stop, side):
        r = abs(entry - stop)
        return entry + r * params["rr_target"] if side == "long" else entry - r * params["rr_target"]

    return simulate_trades(sf, signals, sides, stop_fn, target_fn, capital)


STRATEGIES = {
    "ema_confluence": (run_ema_confluence, EMA_GRID),
    "breakout":       (run_breakout, BREAKOUT_GRID),
    "mean_reversion": (run_mean_reversion, MEANREV_GRID),
    "trend_macd":     (run_trend_macd, TRENDMACD_GRID),
}


_WORKER_TRAIN = None
_WORKER_TEST = None

def _init_worker(train_df, test_df):
    global _WORKER_TRAIN, _WORKER_TEST
    _WORKER_TRAIN = train_df
    _WORKER_TEST = test_df

def _run_combo(packed):
    strategy_name, params, capital = packed
    run_fn, _ = STRATEGIES[strategy_name]
    try:
        train_stats = run_fn(_WORKER_TRAIN, params, capital)
        test_stats = run_fn(_WORKER_TEST, params, capital)
    except Exception:
        return None
    if train_stats is None or test_stats is None:
        return None
    row = {"strategy": strategy_name}
    flat_params = {k: ("+".join(v) if isinstance(v, list) else v) for k, v in params.items()}
    row.update(flat_params)
    for k, v in train_stats.items():
        row[f"train_{k}"] = v
    for k, v in test_stats.items():
        row[f"test_{k}"] = v
    row["robust"] = bool(train_stats["profit_factor"] >= 1.1 and test_stats["profit_factor"] >= 1.0
                          and test_stats["return_pct"] > 0)
    return row


def build_combos(strategy_name: str, grid: dict) -> list:
    keys = list(grid.keys())
    return [dict(zip(keys, c)) for c in itertools.product(*[grid[k] for k in keys])]


def print_snapshot(results: list, done: int, total: int, elapsed: float):
    if not results:
        print(f"  [checkpoint {done}/{total}] elapsed={elapsed/60:.1f}m -- no valid combos yet")
        return
    df = pd.DataFrame(results)
    robust_n = int(df["robust"].sum())
    best_pf = df["test_profit_factor"].max()
    best_row = df.loc[df["test_profit_factor"].idxmax()]
    rate = done / elapsed if elapsed > 0 else 0
    eta_sec = (total - done) / rate if rate > 0 else 0
    print(f"  [checkpoint {done}/{total}] elapsed={elapsed/60:.1f}m eta={eta_sec/60:.1f}m "
          f"valid={len(df)} robust={robust_n} best_test_pf={best_pf:.3f} "
          f"({best_row['strategy']})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="QQQ")
    parser.add_argument("--days", type=int, default=720)
    parser.add_argument("--capital", type=float, default=10000.0)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4,
                         help="Parallel processes (default 4, tuned for 8GB M1)")
    parser.add_argument("--sample", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strategies", default="all")
    args = parser.parse_args()

    print(f"Fetching {args.days}d of 5-min bars for {args.symbol}...")
    df = fetch_bars(args.symbol, args.days)
    if df.empty:
        print("No data fetched - check API keys / symbol."); return
    print(f"Got {len(df)} bars.")

    split_i = int(len(df) * WF_TRAIN_PCT)
    train_df = df.iloc[:split_i]
    test_df = df.iloc[split_i:]
    print(f"Walk-forward split: train={len(train_df)} bars, test={len(test_df)} bars "
          f"({WF_TRAIN_PCT*100:.0f}%/{(1-WF_TRAIN_PCT)*100:.0f}%)")

    names = list(STRATEGIES.keys()) if args.strategies == "all" else [s.strip() for s in args.strategies.split(",")]

    all_combos = []
    for name in names:
        _, grid = STRATEGIES[name]
        combos = build_combos(name, grid)
        if args.sample < 1.0:
            random.seed(args.seed)
            n_sample = max(1, int(len(combos) * args.sample))
            combos = random.sample(combos, n_sample)
        print(f"  {name}: {len(combos)} combos")
        all_combos.extend([(name, c, args.capital) for c in combos])

    total = len(all_combos)
    print(f"\nTotal combos across {len(names)} strategies: {total}")
    print(f"Checkpointing every {CHECKPOINT_EVERY} completed combos (partial CSV + live stats).")
    print(f"Costs: base_slippage={BASE_SLIPPAGE_PCT*100:.3f}% + vol-scaled, "
          f"fill_delay={FILL_DELAY_BARS} bar(s), commission=${COMMISSION_PER_SHARE}/share, "
          f"risk={RISK_PCT*100:.1f}%, cap={MAX_CAPITAL_PCT*100:.0f}%\n")

    out_path = os.path.join(os.getcwd(), "ultimate_sweep_results.csv")
    top_path = os.path.join(os.getcwd(), "ultimate_sweep_top.csv")

    results = []
    done = 0
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                              initargs=(train_df, test_df)) as ex:
        futures = [ex.submit(_run_combo, c) for c in all_combos]
        for fut in as_completed(futures):
            done += 1
            r = fut.result()
            if r is not None:
                results.append(r)

            if done % CHECKPOINT_EVERY == 0 or done == total:
                elapsed = time.time() - start_time
                print_snapshot(results, done, total, elapsed)
                if results:
                    pd.DataFrame(results).to_csv(out_path, index=False)
                    print(f"  -> checkpoint saved to {out_path} ({len(results)} rows so far)\n")

    if not results:
        print("No combo produced enough trades on both splits. Loosen filters or grid.")
        return

    out_df = pd.DataFrame(results)
    out_df.to_csv(out_path, index=False)
    print(f"\nFinal full results ({len(out_df)} rows) saved to {out_path}")

    robust = out_df[out_df["robust"]].copy()
    print(f"Robust (train PF>=1.1 AND test PF>=1.0 AND test return>0): {len(robust)} combos")

    ranked = robust.sort_values("test_profit_factor", ascending=False) if len(robust) else \
             out_df.sort_values("test_return_pct", ascending=False)
    top = ranked.head(args.top)
    top.to_csv(top_path, index=False)
    print(f"Top {len(top)} results saved to {top_path}\n")

    cols = ["strategy", "train_total_trades", "train_win_rate_pct", "train_profit_factor",
            "train_return_pct", "test_total_trades", "test_win_rate_pct", "test_profit_factor",
            "test_return_pct", "robust"]
    print(top[[c for c in cols if c in top.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
