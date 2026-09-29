import pandas as pd
import numpy as np
from datetime import datetime, timezone
from backtester import prepare_strategy_frame, near_ema
from patterns import detect_pattern
from config import StrategyConfig

def run_structural_backtest():
    # 1. Load Data
    # Since I can't access the space-folder easily, I'll try the one the user mentioned
    # or use Alpaca if available. Let's try to read the CSV.
    try:
        # Trying the renamed path or absolute path if known.
        # The user has a file in TRADING_BOT /qqq_5min_1yr.csv
        # I'll try to read it. If it fails, I'll ask for a fix.
        df = pd.read_csv("/Users/batmode/Desktop/TRADING BOT /qqq_5min_1yr.csv", index_col=0, parse_dates=True)
    except Exception as e:
        print(f"Data load failed: {e}")
        return

    # Ensure index is UTC
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')

    # Config for Structural Strategy
    config = StrategyConfig(
        symbol="QQQ",
        timeframe_minutes=5,
        london_timezone="Europe/London",
        bias_start_hour=8,
        bias_start_minute=0,
        no_trade_buffer_minutes=90,
        trend_sma_length=200,
        entry_ema_length=100,
        ema_confluence_pct=0.8,
        atr_length=20,
        atr_stop_buffer_mult=0.2,
        rr_target=2.0,
        enabled_patterns=["liquidity_sweep", "rsi_div", "engulfing", "pin_bar"],
        initial_capital=10000.0,
    )

    sf = prepare_strategy_frame(df, config)

    capital = config.initial_capital
    equity = capital
    trades = []
    in_position = False
    side = entry = stop = target = qty = None

    for i in range(200, len(sf)):
        row = sf.iloc[i]

        if in_position:
            h, l = row["high"], row["low"]
            hit_stop = (side == "long" and l <= stop) or (side == "short" and h >= stop)
            hit_target = (side == "long" and h >= target) or (side == "short" and l <= target)

            if hit_stop or hit_target:
                pnl = (target - entry) if (hit_target and side == "long") or (hit_stop and side == "short") else (stop - entry if side == "long" else entry - stop)
                # Simple PnL calculation
                actual_pnl = (target - entry) * qty if (hit_target and side == "long") or (hit_stop and side == "short") else (stop - entry) * qty if side == "long" else (entry - stop) * qty

                # Let's use a more standard way
                if hit_target:
                    trade_pnl = (target - entry) * qty if side == "long" else (entry - target) * qty
                else:
                    trade_pnl = (stop - entry) * qty if side == "long" else (entry - stop) * qty

                equity += trade_pnl
                trades.append({"won": trade_pnl > 0, "pnl": trade_pnl})
                in_position = False
            continue

        if not row["after_buffer"]: continue

        bias = row["bias"]
        # Structural Logic: Must be near EMA AND have a pattern
        if not near_ema(row, config.ema_confluence_pct): continue

        pattern = detect_pattern(sf, i, bias, config.enabled_patterns)
        if not pattern: continue

        # Entry
        entry_px = row["close"]
        atr = row["atr"]
        if pd.isna(atr): continue

        stop_px = row["low"] - (atr * config.atr_stop_buffer_mult) if bias == "long" else row["high"] + (atr * config.atr_stop_buffer_mult)
        risk = abs(entry_px - stop_px)
        target_px = entry_px + (risk * config.rr_target) if bias == "long" else entry_px - (risk * config.rr_target)

        # Position Sizing (1% risk)
        risk_amt = capital * 0.01
        qty = int(risk_amt // risk) if risk > 0 else 0

        if qty > 0:
            in_position = True
            side = bias
            entry = entry_px
            stop = stop_px
            target = target_px
            qty = qty

    # Results
    if not trades:
        print("No trades executed.")
        return

    win_rate = sum(1 for t in trades if t["won"]) / len(trades) * 100
    net_pnl = equity - capital
    pf = sum(t["pnl"] for t in trades if t["won"]) / abs(sum(t["pnl"] for t in trades if not t["won"])) if any(not t["won"] for t in trades) else 99.0

    print(f"--- Structural Strategy Results ---")
    print(f"Total Trades: {len(trades)}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Net PnL: ${net_pnl:.2f}")
    print(f"Profit Factor: {pf:.3f}")
    print(f"Final Equity: ${equity:.2f}")

if __name__ == "__main__":
    run_structural_backtest()
