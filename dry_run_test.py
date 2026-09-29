"""
SAFE DRY-RUN TEST for paper_bot.py

Exercises the FULL signal -> sizing -> bracket-order-construction pipeline
using REAL recent QQQ market data, WITHOUT ever calling trading_client.submit_order().

Monkey-patches in_regular_session() to True so it works outside market hours.
Prints exactly what order WOULD have been submitted (or why nothing would fire).

Usage:
    source .venv/bin/activate
    python3 dry_run_test.py
"""
import os
import sys

import paper_bot as bot

# ---- 1. Force session check to pass so we can test outside market hours ----
bot.in_regular_session = lambda: True
print("[PATCH] in_regular_session() forced to True for this dry run.\n")


def dry_run():
    config = bot.final_config()
    print(f"[CONFIG] symbol={config.symbol} rr_target={config.rr_target} "
          f"atr_stop_buffer_mult={config.atr_stop_buffer_mult} "
          f"enabled_patterns={config.enabled_patterns}\n")

    trading_client, data_client = bot.get_clients()
    print("[OK] Connected to Alpaca (paper).\n")

    # ---- 2. Real read-only checks (safe: no orders touched) ----
    try:
        pos = bot.get_open_position(trading_client, config.symbol)
        print(f"[STATE] Open position: {pos}" if pos else "[STATE] No open position (flat). \u2705")
    except Exception as e:
        print(f"[ERROR] get_open_position failed: {e}")
        return

    try:
        orders = bot.get_open_orders(trading_client, config.symbol)
        print(f"[STATE] Open orders: {len(orders)}\n")
    except Exception as e:
        print(f"[ERROR] get_open_orders failed: {e}")
        return

    # ---- 3. Fetch REAL recent bars (read-only) ----
    try:
        bars = bot.fetch_recent_bars(data_client, config.symbol)
        print(f"[DATA] Fetched {len(bars)} bars. Latest closed candle: {bars.index[-1]}\n")
    except Exception as e:
        print(f"[ERROR] fetch_recent_bars failed: {e}")
        return

    if len(bars) < 4:
        print("[STOP] Not enough bars to evaluate a signal.")
        return

    latest_complete = bars.iloc[:-1].copy()

    # ---- 4. Run the REAL signal logic ----
    order_signal = bot.signal_to_order(latest_complete, config)

    if order_signal is None:
        print("[RESULT] No valid signal on the latest completed candle right now.")
        print("          (This is normal \u2014 confluence conditions are not always met.)")
        return

    print("[SIGNAL FOUND]")
    for k, v in order_signal.items():
        print(f"    {k}: {v}")
    print()

    # ---- 5. Run the REAL position sizing logic ----
    risk_per_share = abs(order_signal["entry_reference"] - order_signal["stop"])
    risk_pct = float(os.getenv("BOT_RISK_PCT", "0.01"))
    equity = bot.get_account_equity(trading_client)
    risk_amount = equity * risk_pct
    MIN_RISK_PER_SHARE = 0.50
    MAX_CAPITAL_PCT = 0.20
    safe_risk_per_share = max(risk_per_share, MIN_RISK_PER_SHARE)
    qty_by_risk = int(risk_amount // safe_risk_per_share)
    qty_by_capital = int((equity * MAX_CAPITAL_PCT) // order_signal["entry_reference"])
    qty = max(0, min(qty_by_risk, qty_by_capital))

    print("[SIZING]")
    print(f"    equity=${equity:,.2f}  risk_pct={risk_pct}  risk_amount=${risk_amount:.2f}")
    print(f"    risk_per_share={risk_per_share:.4f}  qty_by_risk={qty_by_risk}  "
          f"qty_by_capital={qty_by_capital}  qty_final={qty}")
    print()

    if qty < 1:
        print("[STOP] Computed qty < 1 \u2014 order would be skipped.")
        return

    # ---- 6. Show EXACTLY what bracket order WOULD be submitted \u2014 no submit_order() call ----
    print("=" * 60)
    print("WOULD SUBMIT (dry run \u2014 NOT sent to Alpaca):")
    print("=" * 60)
    print(f"  symbol         = {config.symbol}")
    print(f"  side           = {order_signal['side_label'].upper()}")
    print(f"  qty            = {qty}")
    print(f"  order_class    = BRACKET")
    print(f"  entry_ref      = {order_signal['entry_reference']}")
    print(f"  take_profit    = {order_signal['target']}")
    print(f"  stop_loss      = {order_signal['stop']}")
    print(f"  pattern        = {order_signal['pattern']}")
    print(f"  client_order_id= {order_signal['client_order_id']}")
    print("=" * 60)


if __name__ == "__main__":
    dry_run()
