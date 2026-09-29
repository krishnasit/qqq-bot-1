import os, time
import alpaca_trade_api as tradeapi
from datetime import datetime, timezone

# ── Config ────────────────────────────────────────────────
API_KEY    = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
BASE_URL   = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
SYMBOL     = "QQQ"

api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version="v2")

# ── 1. Flatten any stuck position ────────────────────────
def flatten():
    try:
        pos = api.get_position(SYMBOL)
        qty = int(pos.qty)
        if qty == 0:
            print(f"[OK] No open position in {SYMBOL}")
            return
        side = "sell" if qty > 0 else "buy"
        abs_qty = abs(qty)
        print(f"[WARN] Found stuck {pos.side} position: {qty} shares @ avg {pos.avg_entry_price}")
        order = api.submit_order(
            symbol=SYMBOL, qty=abs_qty,
            side=side, type="market", time_in_force="day"
        )
        print(f"[FIX] Flatten order submitted → {order.id}")
    except Exception as e:
        if "position does not exist" in str(e).lower():
            print(f"[OK] No open position in {SYMBOL}")
        else:
            print(f"[ERROR] {e}")

# ── 2. Cancel all open orders ────────────────────────────
def cancel_all():
    orders = api.list_orders(status="open", symbols=[SYMBOL])
    if not orders:
        print("[OK] No open orders to cancel")
        return
    for o in orders:
        api.cancel_order(o.id)
        print(f"[CANCEL] {o.id} | {o.side} {o.qty} {o.type} @ {o.limit_price or o.stop_price}")

# ── 3. Position guard check (pre-entry) ──────────────────
def is_flat():
    try:
        pos = api.get_position(SYMBOL)
        return int(pos.qty) == 0
    except:
        return True

# ── 4. Account snapshot ──────────────────────────────────
def snapshot():
    acct = api.get_account()
    print(f"\n{'─'*40}")
    print(f"  Equity:        ${float(acct.equity):,.2f}")
    print(f"  Cash:          ${float(acct.cash):,.2f}")
    print(f"  Buying Power:  ${float(acct.buying_power):,.2f}")
    print(f"  Long MV:       ${float(acct.long_market_value):,.2f}")
    print(f"  Short MV:      ${float(acct.short_market_value):,.2f}")
    print(f"  Flat:          {is_flat()}")
    print(f"{'─'*40}\n")

# ── Run ──────────────────────────────────────────────────
print(f"\n[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC] Running position fix...\n")
cancel_all()
flatten()
snapshot()
