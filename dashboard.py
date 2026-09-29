"""
Manual Streamlit dashboard for QQQ strategy testing.
Wraps the same fetch_bars / prepare_strategy_frame / detect_pattern logic
used in single_backtest.py, but exposes EVERY parameter (including
slippage % and commission/share) as manual controls so you can test
one combo at a time in seconds instead of running a multi-hour sweep.

Run with:
    streamlit run dashboard.py
"""
from __future__ import annotations
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from single_backtest import fetch_bars
from backtester import prepare_strategy_frame, near_ema
from patterns import detect_pattern
from config import StrategyConfig

st.set_page_config(page_title="QQQ Strategy Manual Tester", layout="wide")
st.title("QQQ Strategy — Manual Backtest Dashboard")
st.caption("One combo at a time, results in seconds. Slippage/commission/latency already modeled.")

with st.sidebar:
    st.header("Data")
    days = st.number_input("Days of history", min_value=5, max_value=730, value=365, step=5)
    bar_limit = st.number_input("Max bars fetched", min_value=100, max_value=20000, value=5000, step=100)

    st.header("Strategy params")
    entry_ema_length = st.select_slider("Entry EMA length", options=[20, 50, 100], value=100)
    trend_sma_length = st.select_slider("Trend SMA length", options=[100, 200], value=200)
    ema_confluence_pct = st.slider("EMA confluence %", 0.1, 2.0, 0.8, 0.05)
    atr_length = st.select_slider("ATR length", options=[14, 20], value=20)
    atr_stop_buffer_mult = st.slider("ATR stop buffer mult", 0.05, 2.0, 0.1, 0.05)
    rr_target = st.slider("Risk:Reward target", 1.0, 3.0, 2.5, 0.05)
    enabled_patterns = st.multiselect(
        "Enabled patterns",
        ["engulfing", "pin_bar", "liquidity_sweep"],
        default=["engulfing", "pin_bar", "liquidity_sweep"],
    )
    initial_capital = st.number_input("Initial capital ($)", min_value=100.0, value=10000.0, step=500.0)
    risk_pct = st.slider("Risk per trade (%)", 0.1, 3.0, 1.0, 0.1)
    max_capital_pct = st.slider("Max capital per trade (%)", 5.0, 100.0, 20.0, 5.0)

    st.header("Real-world costs")
    slippage_pct = st.slider("Slippage per fill (%)", 0.0, 0.5, 0.05, 0.01) / 100.0
    commission_per_share = st.number_input(
        "Commission per share ($)", min_value=0.0, max_value=0.05, value=0.0002, step=0.0001, format="%.4f"
    )
    latency_bars = st.slider("Entry latency (bars, 5-min each)", 0, 5, 1, 1)

    run_btn = st.button("Run backtest", type="primary")


def run_manual_backtest(df, cfg, slippage_pct, commission_per_share, latency_bars, risk_pct, max_capital_pct, record_trades=True):
    frame = prepare_strategy_frame(df, cfg)
    capital = float(cfg.initial_capital)
    equity = capital
    peak = capital
    max_dd = 0.0
    wins = losses = 0
    total_r = 0.0
    gross_win = gross_loss = 0.0
    trade_log = []
    equity_curve = []
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
                raw_exit_price = stop if hit_stop else target
                exit_price = raw_exit_price * (1 - slippage_pct) if side == "long" else raw_exit_price * (1 + slippage_pct)
                r_mult = -1.0 if hit_stop else float(cfg.rr_target)
                risk_per_share_at_entry = abs(entry - stop)
                pnl = trade_qty * risk_per_share_at_entry * r_mult
                pnl -= trade_qty * commission_per_share * 2
                equity += pnl
                gross_win += max(pnl, 0)
                gross_loss += abs(min(pnl, 0))
                total_r += r_mult
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1
                peak = max(peak, equity)
                dd = (peak - equity) / peak * 100 if peak > 0 else 0
                max_dd = max(max_dd, dd)
                if record_trades:
                    trade_log.append({
                        "entry_time": entry_ts, "exit_time": ts, "side": side,
                        "entry_price": round(entry, 4), "exit_price": round(exit_price, 4),
                        "stop_price": round(stop, 4), "target_price": round(target, 4),
                        "r_multiple": r_mult, "pnl_usd": round(pnl, 2),
                        "result": "WIN" if pnl > 0 else "LOSS",
                        "equity": round(equity, 2), "drawdown_pct": round(dd, 2),
                    })
                equity_curve.append((ts, equity))
                in_position = False
            continue

        if not bool(row.get("after_buffer", False)):
            continue
        bias = row.get("bias")
        if pd.isna(bias) or bias is None:
            continue
        if not near_ema(row, cfg.ema_confluence_pct):
            continue
        pattern = detect_pattern(frame, i, bias, cfg.enabled_patterns)
        if not pattern:
            continue
        if pd.isna(row.get("atr")):
            continue

        entry_idx = i + max(1, latency_bars)
        if entry_idx >= n:
            continue
        next_row = frame.iloc[entry_idx]
        next_ts = frame.index[entry_idx]

        signal_close = float(row["close"])
        raw_entry_price = float(next_row["open"])
        entry_price = raw_entry_price * (1 + slippage_pct) if bias == "long" else raw_entry_price * (1 - slippage_pct)
        atr_buffer = float(row["atr"]) * float(cfg.atr_stop_buffer_mult)
        min_buffer = max(0.05, signal_close * 0.0005)
        effective_buffer = max(atr_buffer, min_buffer)

        if bias == "long":
            stop_price = float(row["low"]) - effective_buffer
            target_price = entry_price + (entry_price - stop_price) * float(cfg.rr_target)
        else:
            stop_price = float(row["high"]) + effective_buffer
            target_price = entry_price - (stop_price - entry_price) * float(cfg.rr_target)

        risk_per_share = abs(entry_price - stop_price)
        if risk_per_share <= 0:
            continue
        risk_dollars = equity * (risk_pct / 100.0)
        qty_by_risk = risk_dollars / risk_per_share
        qty_by_cap = (equity * (max_capital_pct / 100.0)) / entry_price
        trade_qty = max(0.0, min(qty_by_risk, qty_by_cap))
        if trade_qty <= 0:
            continue

        side = bias
        entry = entry_price
        stop = stop_price
        target = target_price
        entry_ts = next_ts
        in_position = True

    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
    total_trades = wins + losses
    win_rate = (wins / total_trades * 100) if total_trades else 0.0
    total_return_pct = (equity - capital) / capital * 100

    stats = {
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(profit_factor, 3) if profit_factor != float("inf") else "inf",
        "avg_r": round(total_r / total_trades, 3) if total_trades else 0.0,
        "return_pct": round(total_return_pct, 2),
        "max_dd_pct": round(max_dd, 2),
        "final_equity": round(equity, 2),
        "net_pnl": round(equity - capital, 2),
    }
    return stats, pd.DataFrame(trade_log), pd.DataFrame(equity_curve, columns=["time", "equity"])


if run_btn:
    with st.spinner("Fetching bars and running backtest..."):
        df = fetch_bars("QQQ", int(days), limit=int(bar_limit))
        cfg = StrategyConfig(
            entry_ema_length=int(entry_ema_length),
            trend_sma_length=int(trend_sma_length),
            ema_confluence_pct=float(ema_confluence_pct),
            atr_length=int(atr_length),
            atr_stop_buffer_mult=float(atr_stop_buffer_mult),
            rr_target=float(rr_target),
            enabled_patterns=enabled_patterns,
            initial_capital=float(initial_capital),
        )
        stats, trades_df, equity_df = run_manual_backtest(
            df, cfg, slippage_pct, commission_per_share, int(latency_bars),
            float(risk_pct), float(max_capital_pct),
        )

    st.subheader("Results")
    cols = st.columns(4)
    cols[0].metric("Trades", stats["total_trades"])
    cols[1].metric("Win rate", f"{stats['win_rate_pct']}%")
    cols[2].metric("Profit factor", stats["profit_factor"])
    cols[3].metric("Max DD", f"{stats['max_dd_pct']}%")

    cols2 = st.columns(4)
    cols2[0].metric("Return", f"{stats['return_pct']}%")
    cols2[1].metric("Net P&L", f"${stats['net_pnl']}")
    cols2[2].metric("Avg R", stats["avg_r"])
    cols2[3].metric("Final equity", f"${stats['final_equity']}")

    if not equity_df.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=equity_df["time"], y=equity_df["equity"], mode="lines", name="Equity"))
        fig.update_layout(title="Equity curve", xaxis_title="Time", yaxis_title="Equity ($)", height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Trade log")
    st.dataframe(trades_df, use_container_width=True)

    csv = trades_df.to_csv(index=False)
    st.download_button("Download trade log CSV", csv, file_name="manual_trade_log.csv")
else:
    st.info("Set parameters in the sidebar (slippage/commission/latency included) and click **Run backtest**.")
