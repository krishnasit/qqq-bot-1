from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from alpaca.data.enums import DataFeed
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import MarketOrderRequest, StopLossRequest, TakeProfitRequest

from backtester import near_ema, prepare_strategy_frame
from config import StrategyConfig
from patterns import detect_pattern

import sys
import structlog

load_dotenv()

logging.basicConfig(
    format="%(message)s",
    stream=sys.stdout,
    level=logging.INFO,
)

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.EventRenamer("message"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    cache_logger_on_first_use=True,
)

LOG = structlog.get_logger("paper_bot")


def final_config() -> StrategyConfig:
    return StrategyConfig(
        symbol=os.getenv("BOT_SYMBOL", "QQQ"),
        timeframe_minutes=5,
        london_timezone="Europe/London",
        bias_start_hour=8,
        bias_start_minute=0,
        no_trade_buffer_minutes=90,
        trend_sma_length=200,
        entry_ema_length=50,
        ema_confluence_pct=0.8,
        atr_length=20,
        atr_stop_buffer_mult=0.25,
        rr_target=1.5,
        trailing_stop_enabled=False,
        trailing_atr_mult=1.0,
        enabled_patterns=["engulfing"],
        initial_capital=10000.0,
    )


def state_path() -> Path:
    return Path(os.getenv("BOT_STATE_PATH", "paper_bot_state.json"))


def load_state() -> dict:
    path = state_path()
    if not path.exists():
        return {"loss_streak": 0, "last_loss_order_id": "", "last_trade_day": "", "trades_today": 0}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {"loss_streak": 0, "last_loss_order_id": "", "last_trade_day": "", "trades_today": 0}


def save_state(state: dict) -> None:
    state_path().write_text(json.dumps(state, indent=2, sort_keys=True))


def today_key() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def reset_daily_counters_if_needed(state: dict) -> dict:
    today = today_key()
    if state.get("last_trade_day") != today:
        state["last_trade_day"] = today
        state["trades_today"] = 0
    return state


def get_clients():
    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("Missing ALPACA_API_KEY or ALPACA_SECRET_KEY in environment/.env")

    paper_flag = os.getenv("ALPACA_PAPER", "true").strip().lower()
    if paper_flag not in {"1", "true", "yes", "on"}:
        raise RuntimeError("This script is intentionally paper-only. Set ALPACA_PAPER=true.")

    trading = TradingClient(key, secret, paper=True)
    market_data = StockHistoricalDataClient(key, secret)
    return trading, market_data


def get_account_equity(trading_client) -> float:
    account = trading_client.get_account()
    return float(account.equity)


def in_regular_session() -> bool:
    now = datetime.now(timezone.utc)
    minutes = now.hour * 60 + now.minute
    return now.weekday() < 5 and (13 * 60 + 30) <= minutes < (20 * 60)


def fetch_recent_bars(data_client, symbol: str, lookback_days: int = 10, limit: int = 500) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - pd.Timedelta(days=lookback_days).to_pytimedelta()

    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        start=start,
        end=end,
        feed=DataFeed.IEX,
    )
    bars = data_client.get_stock_bars(req).df

    if bars is None or bars.empty:
        raise RuntimeError("No recent bars returned from Alpaca.")

    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.reset_index()

    if "symbol" in bars.columns:
        bars = bars[bars["symbol"] == symbol].copy()

    out = bars[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = out.set_index("timestamp").sort_index().tail(limit)
    return out


def get_open_orders(trading_client, symbol: str):
    from alpaca.trading.requests import GetOrdersRequest
    request = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=100)
    orders = trading_client.get_orders(filter=request)
    return [o for o in orders if getattr(o, "symbol", None) == symbol]


def get_open_position(trading_client, symbol: str):
    try:
        return trading_client.get_open_position(symbol)
    except Exception as exc:
        msg = str(exc).lower()
        if "position does not exist" in msg or "not found" in msg or "404" in msg:
            return None
        raise


def _safe_attr(obj, *names, default=None):
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return default


def _to_float(value):
    try:
        return float(value)
    except Exception:
        return None


def _normalize_side(value) -> str:
    if value is None:
        return ""
    s = str(value).lower()
    if "buy" in s:
        return "buy"
    if "sell" in s:
        return "sell"
    return s


def get_recent_closed_orders(trading_client, symbol: str, days_back: int = 5):
    after = (datetime.now(timezone.utc) - timedelta(days=days_back)).isoformat()
    orders = []
    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus

        try:
            req = GetOrdersRequest(
                status=QueryOrderStatus.CLOSED,
                after=after,
                limit=200,
                nested=True,
            )
        except TypeError:
            req = GetOrdersRequest(
                status=QueryOrderStatus.CLOSED,
                limit=200,
                nested=True,
            )

        orders = trading_client.get_orders(filter=req)
    except Exception:
        try:
            orders = trading_client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=200))
        except Exception:
            LOG.warning("Could not fetch closed orders from Alpaca; loss-streak sync skipped for this cycle.")
            return []

    return [o for o in orders if getattr(o, "symbol", None) == symbol]


def extract_completed_trade_outcomes(closed_orders):
    entries = []
    exits = []

    for order in closed_orders:
        oid = str(_safe_attr(order, "id", default="") or "")
        side = _normalize_side(_safe_attr(order, "side", default=""))
        status = str(_safe_attr(order, "status", default="") or "").lower()
        client_order_id = str(_safe_attr(order, "client_order_id", default="") or "")
        filled_avg_price = _to_float(_safe_attr(order, "filled_avg_price", "filled_avg", default=None))
        submitted_at = str(_safe_attr(order, "submitted_at", "created_at", "filled_at", default="") or "")
        order_class = str(_safe_attr(order, "order_class", default="") or "").lower()
        parent_order_id = str(_safe_attr(order, "parent_order_id", default="") or "")

        if status not in {"filled"}:
            continue

        rec = {
            "id": oid,
            "side": side,
            "filled_avg_price": filled_avg_price,
            "submitted_at": submitted_at,
            "client_order_id": client_order_id,
            "order_class": order_class,
            "parent_order_id": parent_order_id,
        }

        if order_class == "bracket" or client_order_id.startswith("qlb-"):
            entries.append(rec)
        elif parent_order_id:
            exits.append(rec)

    exits_by_parent = {}
    for ex in exits:
        exits_by_parent.setdefault(ex["parent_order_id"], []).append(ex)

    completed = []
    for entry in entries:
        matches = exits_by_parent.get(entry["id"], [])
        if not matches:
            continue
        entry_price = entry["filled_avg_price"]
        if entry_price is None:
            continue

        exit_order = sorted(matches, key=lambda x: x["submitted_at"])[-1]
        exit_price = exit_order["filled_avg_price"]
        if exit_price is None:
            continue

        if entry["side"] == "buy":
            pnl = exit_price - entry_price
            won = pnl > 0
            direction = "long"
        elif entry["side"] == "sell":
            pnl = entry_price - exit_price
            won = pnl > 0
            direction = "short"
        else:
            continue

        completed.append(
            {
                "entry_order_id": entry["id"],
                "entry_client_order_id": entry["client_order_id"],
                "direction": direction,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "won": won,
                "pnl_per_share": pnl,
                "submitted_at": entry["submitted_at"],
            }
        )

    completed.sort(key=lambda x: x["submitted_at"])
    return completed


def update_loss_streak_from_broker(trading_client, symbol: str, state: dict) -> dict:
    closed_orders = get_recent_closed_orders(trading_client, symbol)
    completed = extract_completed_trade_outcomes(closed_orders)

    if not completed:
        return state

    latest = completed[-1]
    latest_key = latest["entry_order_id"] or latest["entry_client_order_id"]
    if not latest_key:
        return state

    if state.get("last_loss_order_id") == latest_key:
        return state

    streak = 0
    for trade in reversed(completed):
        if trade["won"]:
            break
        streak += 1

    state["loss_streak"] = streak
    state["last_loss_order_id"] = latest_key
    return state


def signal_to_order(latest_complete_df: pd.DataFrame, config: StrategyConfig):
    frame = prepare_strategy_frame(latest_complete_df, config)

    if len(frame) < 3:
        return None

    i = len(frame) - 1
    row = frame.iloc[i]
    ts = frame.index[i]

    if not bool(row.get("after_buffer", False)):
        return None

    bias = row.get("bias")
    if pd.isna(bias) or bias is None:
        return None

    if not near_ema(row, config.ema_confluence_pct):
        return None

    pattern = detect_pattern(frame, i, bias, config.enabled_patterns)
    if not pattern:
        return None

    if pd.isna(row.get("atr")):
        return None

    entry = float(row["close"])
    atr_buffer = float(row["atr"]) * float(config.atr_stop_buffer_mult)

    if bias == "long":
        stop = float(row["low"]) - atr_buffer
        target = entry + (entry - stop) * float(config.rr_target)
        side = OrderSide.BUY
        side_label = "long"
    else:
        stop = float(row["high"]) + atr_buffer
        target = entry - (stop - entry) * float(config.rr_target)
        side = OrderSide.SELL
        side_label = "short"

    risk = abs(entry - stop)
    if risk <= 0 or stop <= 0 or target <= 0:
        return None

    raw = f"{config.symbol}:{side_label}:{ts.isoformat()}".encode()
    client_order_id = "qlb-" + hashlib.sha256(raw).hexdigest()[:24]

    return {
        "signal_timestamp": ts,
        "symbol": config.symbol,
        "side": side,
        "side_label": side_label,
        "entry_reference": round(entry, 4),
        "stop": round(stop, 2),
        "target": round(target, 2),
        "pattern": pattern,
        "client_order_id": client_order_id,
    }


def append_log(csv_path: str, row: dict):
    path = Path(csv_path)
    fields = [
        "logged_at",
        "event",
        "symbol",
        "signal_timestamp",
        "side",
        "entry_reference",
        "stop",
        "target",
        "pattern",
        "client_order_id",
        "broker_order_id",
        "message",
    ]

    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fields})


def execute_once() -> int:
    config = final_config()
    trading_client, data_client = get_clients()
    log_path = os.getenv("BOT_LOG_PATH", "paper_trades.csv")
    max_consecutive_losses = int(os.getenv("BOT_MAX_CONSECUTIVE_LOSSES", "3"))
    max_trades_per_day = int(os.getenv("BOT_MAX_TRADES_PER_DAY", "1"))

    state = load_state()
    state = reset_daily_counters_if_needed(state)
    state = update_loss_streak_from_broker(trading_client, config.symbol, state)
    save_state(state)

    if not in_regular_session():
        LOG.info("Outside regular US session. No action taken.")
        return 0

    if state.get("loss_streak", 0) >= max_consecutive_losses:
        LOG.warning(
            "Loss streak lock active for %s: %s consecutive losses. No new order sent.",
            config.symbol,
            state.get("loss_streak", 0),
        )
        return 0

    if state.get("trades_today", 0) >= max_trades_per_day:
        LOG.info(
            "Daily trade cap reached for %s: %s trades today. No new order sent.",
            config.symbol,
            state.get("trades_today", 0),
        )
        return 0

    open_position = get_open_position(trading_client, config.symbol)
    open_orders = get_open_orders(trading_client, config.symbol)

    if open_position is not None:
        LOG.info("Existing open position detected for %s. No new order sent.", config.symbol)
        return 0

    if open_orders:
        LOG.info("Existing open order(s) detected for %s. No new order sent.", config.symbol)
        return 0

    bars = fetch_recent_bars(data_client, config.symbol)
    if len(bars) < 4:
        LOG.info("Not enough bars yet.")
        return 0

    latest_complete = bars.iloc[:-1].copy()
    order_signal = signal_to_order(latest_complete, config)

    if order_signal is None:
        LOG.info("No valid signal on the latest completed candle.")
        return 0

    risk_per_share = abs(order_signal["entry_reference"] - order_signal["stop"])
    if risk_per_share <= 0:
        LOG.warning("Invalid risk_per_share (%s). Skipping order.", risk_per_share)
        return 0

    risk_pct = float(os.getenv("BOT_RISK_PCT", "0.01"))
    equity = get_account_equity(trading_client)
    risk_amount = equity * risk_pct
    qty = int(risk_amount // risk_per_share)

    if qty < 1:
        LOG.warning(
            "Computed qty < 1 (equity=%.2f, risk_pct=%.4f, risk_per_share=%.4f). Skipping order.",
            equity, risk_pct, risk_per_share,
        )
        return 0

    LOG.info(
        "Position sizing: equity=%.2f risk_pct=%.4f risk_amount=%.2f risk_per_share=%.4f qty=%s",
        equity, risk_pct, risk_amount, risk_per_share, qty,
    )

    req = MarketOrderRequest(
        symbol=config.symbol,
        qty=qty,
        side=order_signal["side"],
        time_in_force=TimeInForce.DAY,
        order_class=OrderClass.BRACKET,
        take_profit=TakeProfitRequest(limit_price=order_signal["target"]),
        stop_loss=StopLossRequest(stop_price=order_signal["stop"]),
        client_order_id=order_signal["client_order_id"],
    )

    order = trading_client.submit_order(req)

    state["trades_today"] = int(state.get("trades_today", 0)) + 1
    state["last_trade_day"] = today_key()
    save_state(state)

    log_row = {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "event": "bracket_order_submitted",
        "symbol": config.symbol,
        "signal_timestamp": order_signal["signal_timestamp"].isoformat(),
        "side": order_signal["side_label"],
        "entry_reference": order_signal["entry_reference"],
        "stop": order_signal["stop"],
        "target": order_signal["target"],
        "pattern": order_signal["pattern"],
        "client_order_id": order_signal["client_order_id"],
        "broker_order_id": str(getattr(order, "id", "")),
        "message": "Paper bracket order submitted",
    }
    append_log(log_path, log_row)
    LOG.warning("PAPER ORDER SUBMITTED: %s", log_row)
    return 1


def main():
    parser = argparse.ArgumentParser(description="QQQ London Bias Alpaca paper bot")
    parser.add_argument("--once", action="store_true", help="Run one evaluation cycle")
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    args = parser.parse_args()


    if args.once == args.loop:
        parser.error("Choose exactly one of --once or --loop")

    if args.once:
        execute_once()
        return

    poll_seconds = max(30, int(os.getenv("BOT_POLL_SECONDS", "60")))
    while True:
        try:
            execute_once()
        except Exception:
            LOG.exception("Paper bot cycle failed.")
        time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
