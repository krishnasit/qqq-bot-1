"""
Quick core-parameter sweep — finishes in ~10-20 min.
Fetches bars ONCE, then tests combos of the main parameters in parallel.
Same realistic slippage/commission/entry-lag as single_backtest.py.
Run with: python3 quick_sweep.py
"""
from __future__ import annotations
import itertools, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd

from single_backtest import fetch_bars
from backtester import prepare_strategy_frame, near_ema
from patterns import detect_pattern
from config import StrategyConfig

SLIPPAGE_PCT = 0.0005
COMMISSION_PER_SHARE = 0.0002
LATENCY_BARS = 1
RISK_PCT = 1.0
MAX_CAPITAL_PCT = 20.0
INITIAL_CAPITAL = 10000.0
DAYS = 365
BAR_LIMIT = 5000

# Core parameter grid — kept small on purpose for a fast sweep
ENTRY_EMA = [20, 50, 100]
TREND_SMA = [100, 200]
EMA_CONFLUENCE = [0.2, 0.5, 0.8]
ATR_LENGTH = [14, 20]
ATR_BUFFER = [0.1, 0.3]
RR_TARGET = [1.5, 2.0, 2.5, 3.0]
PATTERNS = ["engulfing", "pin_bar", "liquidity_sweep"]


def run_one(args):
    entry_ema, trend_sma, ema_conf, atr_len, atr_buf, rr, frame_pickle = args
    frame = pd.read_pickle(frame_pickle) if isinstance(frame_pickle, str) else frame_pickle
    cfg = StrategyConfig(
        entry_ema_length=entry_ema, trend_sma_length=trend_sma,
        ema_confluence_pct=ema_conf, atr_length=atr_len,
        atr_stop_buffer_mult=atr_buf, rr_target=rr,
        enabled_patterns=PATTERNS, initial_capital=INITIAL_CAPITAL,
    )
    frame2 = prepare_strategy_frame(frame, cfg)
    capital = INITIAL_CAPITAL
    equity = capital
    peak = capital
    max_dd = 0.0
    wins = losses = 0
    total_r = 0.0
    gross_win = gross_loss = 0.0
    in_position = False
    side = entry = stop = target = None
    n = len(frame2)

    for i in range(2, n):
        row = frame2.iloc[i]
        if in_position:
            hit_stop = (side == "long" and row["low"] <= stop) or (side == "short" and row["high"] >= stop)
            hit_target = (side == "long" and row["high"] >= target) or (side == "short" and row["low"] <= target)
            if hit_stop or hit_target:
                r_mult = -1.0 if hit_stop else rr
                risk_per_share = abs(entry - stop)
                pnl = trade_qty * risk_per_share * r_mult
                pnl -= trade_qty * COMMISSION_PER_SHARE * 2
                equity += pnl
                gross_win += max(pnl, 0); gross_loss += abs(min(pnl, 0))
                total_r += r_mult
                if pnl > 0: wins += 1
                else: losses += 1
                peak = max(peak, equity)
                dd = (peak - equity) / peak * 100 if peak > 0 else 0
                max_dd = max(max_dd, dd)
                in_position = False
            continue

        if not bool(row.get("after_buffer", False)): continue
        bias = row.get("bias")
        if pd.isna(bias) or bias is None: continue
        if not near_ema(row, cfg.ema_confluence_pct): continue
        pattern = detect_pattern(frame2, i, bias, cfg.enabled_patterns)
        if not pattern: continue
        if pd.isna(row.get("atr")): continue

        entry_idx = i + LATENCY_BARS
        if entry_idx >= n: continue
        next_row = frame2.iloc[entry_idx]
        signal_close = float(row["close"])
        raw_entry = float(next_row["open"])
        entry_price = raw_entry * (1 + SLIPPAGE_PCT) if bias == "long" else raw_entry * (1 - SLIPPAGE_PCT)
        atr_buf_val = float(row["atr"]) * atr_buf
        min_buf = max(0.05, signal_close * 0.0005)
        eff_buf = max(atr_buf_val, min_buf)

        if bias == "long":
            stop_price = float(row["low"]) - eff_buf
            target_price = entry_price + (entry_price - stop_price) * rr
        else:
            stop_price = float(row["high"]) + eff_buf
            target_price = entry_price - (stop_price - entry_price) * rr

        risk_per_share = abs(entry_price - stop_price)
        if risk_per_share <= 0: continue
        risk_dollars = equity * (RISK_PCT / 100.0)
        qty_by_risk = risk_dollars / risk_per_share
        qty_by_cap = (equity * (MAX_CAPITAL_PCT / 100.0)) / entry_price
        trade_qty = max(0.0, min(qty_by_risk, qty_by_cap))
        if trade_qty <= 0: continue

        side, entry, stop, target = bias, entry_price, stop_price, target_price
        in_position = True

    total_trades = wins + losses
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
    win_rate = (wins / total_trades * 100) if total_trades else 0.0
    return {
        "entry_ema_length": entry_ema, "trend_sma_length": trend_sma,
        "ema_confluence_pct": ema_conf, "atr_length": atr_len,
        "atr_stop_buffer_mult": atr_buf, "rr_target": rr,
        "total_trades": total_trades, "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(profit_factor, 3) if profit_factor != float("inf") else 999,
        "return_pct": round((equity - capital) / capital * 100, 2),
        "max_dd_pct": round(max_dd, 2), "net_pnl": round(equity - capital, 2),
        "avg_r": round(total_r / total_trades, 3) if total_trades else 0.0,
    }


if __name__ == "__main__":
    t0 = time.time()
    print("Fetching bars once...")
    df = fetch_bars("QQQ", DAYS, limit=BAR_LIMIT)
    print(f"Fetched {len(df)} bars in {time.time()-t0:.1f}s")

    combos = list(itertools.product(ENTRY_EMA, TREND_SMA, EMA_CONFLUENCE, ATR_LENGTH, ATR_BUFFER, RR_TARGET))
    print(f"Running {len(combos)} combos...")

    tasks = [(e, s, c, al, ab, rr, df) for e, s, c, al, ab, rr in combos]
    results = []
    with ProcessPoolExecutor() as ex:
        futures = [ex.submit(run_one, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            results.append(fut.result())
            if i % 50 == 0:
                print(f"  {i}/{len(combos)} done ({time.time()-t0:.0f}s elapsed)")

    out = pd.DataFrame(results).sort_values(["profit_factor", "return_pct"], ascending=False)
    out.to_csv("quick_sweep_results.csv", index=False)
    print(f"\nDone in {time.time()-t0:.1f}s. Saved quick_sweep_results.csv")
    print("\nTop 10 by profit factor:")
    print(out.head(10).to_string(index=False))
