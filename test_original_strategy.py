"""
Tests the ORIGINAL 3-factor strategy: 50 EMA + 200 SMA + VWAP confluence,
London session bias (90-min buffer), ATR-based SL/TP.
Adds VWAP (missing from backtester.py) as a third confirmation factor.
Realistic slippage/commission/entry-lag included.
Run with: python3 test_original_strategy.py
"""
from __future__ import annotations
import pandas as pd
from single_backtest import fetch_bars
from backtester import prepare_strategy_frame, near_ema, _compute_after_buffer
from patterns import detect_pattern
from config import StrategyConfig

SLIPPAGE_PCT = 0.0005
COMMISSION_PER_SHARE = 0.0002
LATENCY_BARS = 1
RISK_PCT = 1.0
MAX_CAPITAL_PCT = 20.0
INITIAL_CAPITAL = 10000.0


def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = out.index.tz_convert("Europe/London").date
    typical = (out["high"] + out["low"] + out["close"]) / 3.0
    cum_vol = out.groupby("date")["volume"].cumsum()
    cum_tp_vol = (typical * out["volume"]).groupby(out["date"]).cumsum()
    out["vwap"] = cum_tp_vol / cum_vol
    return out.drop(columns=["date"])


def vwap_confirms(row) -> bool:
    close = row.get("close")
    vwap = row.get("vwap")
    bias = row.get("bias")
    if pd.isna(close) or pd.isna(vwap) or bias is None:
        return False
    if bias == "long":
        return close > vwap
    if bias == "short":
        return close < vwap
    return False


def run_backtest(df, cfg, ema_confluence_pct, record_trades=True):
    frame = prepare_strategy_frame(df, cfg)
    frame = add_vwap(frame)
    capital = INITIAL_CAPITAL
    equity = capital
    peak = capital
    max_dd = 0.0
    wins = losses = 0
    total_r = 0.0
    gross_win = gross_loss = 0.0
    trade_log = []
    in_position = False
    side = entry = stop = target = entry_ts = None
    trade_qty = 0.0
    n = len(frame)

    for i in range(2, n):
        row = frame.iloc[i]
        ts = frame.index[i]
        if in_position:
            hit_stop = (side == "long" and row["low"] <= stop) or (side == "short" and row["high"] >= stop)
            hit_target = (side == "long" and row["high"] >= target) or (side == "short" and row["low"] <= target)
            if hit_stop or hit_target:
                r_mult = -1.0 if hit_stop else float(cfg.rr_target)
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
                if record_trades:
                    trade_log.append({"entry_time": entry_ts, "exit_time": ts, "side": side,
                        "pnl_usd": round(pnl, 2), "result": "WIN" if pnl > 0 else "LOSS",
                        "r_multiple": r_mult, "equity": round(equity, 2)})
                in_position = False
            continue

        if not bool(row.get("after_buffer", False)): continue
        bias = row.get("bias")
        if pd.isna(bias) or bias is None: continue
        if not near_ema(row, ema_confluence_pct): continue
        if not vwap_confirms(row): continue
        pattern = detect_pattern(frame, i, bias, cfg.enabled_patterns)
        if not pattern: continue
        if pd.isna(row.get("atr")): continue

        entry_idx = i + LATENCY_BARS
        if entry_idx >= n: continue
        next_row = frame.iloc[entry_idx]
        next_ts = frame.index[entry_idx]
        signal_close = float(row["close"])
        raw_entry = float(next_row["open"])
        entry_price = raw_entry * (1 + SLIPPAGE_PCT) if bias == "long" else raw_entry * (1 - SLIPPAGE_PCT)
        atr_buf = float(row["atr"]) * float(cfg.atr_stop_buffer_mult)
        min_buf = max(0.05, signal_close * 0.0005)
        eff_buf = max(atr_buf, min_buf)

        if bias == "long":
            stop_price = float(row["low"]) - eff_buf
            target_price = entry_price + (entry_price - stop_price) * float(cfg.rr_target)
        else:
            stop_price = float(row["high"]) + eff_buf
            target_price = entry_price - (stop_price - entry_price) * float(cfg.rr_target)

        risk_per_share = abs(entry_price - stop_price)
        if risk_per_share <= 0: continue
        risk_dollars = equity * (RISK_PCT / 100.0)
        qty_by_risk = risk_dollars / risk_per_share
        qty_by_cap = (equity * (MAX_CAPITAL_PCT / 100.0)) / entry_price
        trade_qty = max(0.0, min(qty_by_risk, qty_by_cap))
        if trade_qty <= 0: continue

        side, entry, stop, target, entry_ts = bias, entry_price, stop_price, target_price, next_ts
        in_position = True

    total_trades = wins + losses
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
    win_rate = (wins / total_trades * 100) if total_trades else 0.0
    stats = {
        "total_trades": total_trades, "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(profit_factor, 3) if profit_factor != float("inf") else 999,
        "return_pct": round((equity - capital) / capital * 100, 2),
        "max_dd_pct": round(max_dd, 2), "net_pnl": round(equity - capital, 2),
        "avg_r": round(total_r / total_trades, 3) if total_trades else 0.0,
    }
    return stats, pd.DataFrame(trade_log)


if __name__ == "__main__":
    df = fetch_bars("QQQ", 365, limit=5000)
    print(f"Fetched {len(df)} bars\n")

    for confluence in [0.15, 0.20]:
        cfg = StrategyConfig(
            entry_ema_length=50, trend_sma_length=200,
            ema_confluence_pct=confluence, atr_length=20,
            atr_stop_buffer_mult=0.1, rr_target=2.5,
            no_trade_buffer_minutes=90,
            enabled_patterns=["engulfing", "pin_bar", "liquidity_sweep"],
        )
        stats, trades_df = run_backtest(df, cfg, confluence)
        print(f"=== EMA confluence {confluence}% ===")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        trades_df.to_csv(f"original_strategy_trades_{confluence}.csv", index=False)
        print()
