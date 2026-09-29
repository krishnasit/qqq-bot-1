from __future__ import annotations
import argparse, itertools, os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
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

# ── Real-world cost constants ──────────────────────────────────────────────
SLIPPAGE_PCT         = 0.0005    # 0.05% adverse slippage per fill (entry & exit)
COMMISSION_PER_SHARE = 0.0002    # Alpaca real cost: $0 commission + SEC/FINRA sell-side only
RISK_PCT             = 0.015     # 1.5% risk per trade target
MAX_CAPITAL_PCT      = 0.25      # max 25% of equity per position (safety cap)
MIN_RISK_PER_SHARE   = 0.50      # floor on stop distance

# ── Expanded parameter grid ────────────────────────────────────────────────
PARAM_GRID = {
    "entry_ema_length":        [10, 15, 20, 30, 50, 100],
    "trend_sma_length":        [50, 100, 150, 200],
    "ema_confluence_pct":      [0.5, 0.8, 1.0, 1.2, 1.5, 2.0],
    "atr_length":              [10, 14, 20, 25, 30],
    "atr_stop_buffer_mult":    [0.1, 0.2, 0.25, 0.4, 0.5],
    "rr_target":               [1.1, 1.2, 1.3, 1.5, 1.75, 2.0, 2.2, 2.5],
    "no_trade_buffer_minutes": [30, 60, 90],
    "enabled_patterns": [
        ["engulfing"],
        ["pin_bar"],
        ["liquidity_sweep"],
        ["engulfing", "pin_bar"],
        ["engulfing", "liquidity_sweep"],
        ["engulfing", "pin_bar", "liquidity_sweep"],
        ["engulfing", "pin_bar", "wick_rejection"],
        ["engulfing", "pin_bar", "inside_bar"],
    ],
}


def fetch_bars(symbol: str, days: int) -> pd.DataFrame:
    api_key    = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not api_secret:
        raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_SECRET_KEY in .env")
    client = StockHistoricalDataClient(api_key, api_secret)
    end    = datetime.now(timezone.utc)
    start  = end - timedelta(days=days)
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


def run_backtest(df: pd.DataFrame, config: StrategyConfig) -> dict | None:
    sf = prepare_strategy_frame(df, config)
    capital     = float(config.initial_capital)
    equity      = capital
    peak        = capital
    max_dd      = 0.0
    wins        = losses = 0
    gross_win   = gross_loss = 0.0
    total_r     = 0.0
    in_position = False
    side = entry = stop = target = trade_qty = None
    warmup = max(config.trend_sma_length, config.atr_length, config.entry_ema_length, 15)

    for i in range(warmup, len(sf)):
        row = sf.iloc[i]

        if in_position:
            h, l = float(row["high"]), float(row["low"])
            hit_stop   = (side == "long"  and l <= stop)  or (side == "short" and h >= stop)
            hit_target = (side == "long"  and h >= target) or (side == "short" and l <= target)
            if hit_stop or hit_target:
                raw_exit   = stop if hit_stop else target
                exit_price = raw_exit * (1 - SLIPPAGE_PCT) if side == "long" else raw_exit * (1 + SLIPPAGE_PCT)
                r_mult     = -1.0 if hit_stop else float(config.rr_target)
                pnl        = trade_qty * abs(entry - stop) * r_mult
                pnl       -= trade_qty * COMMISSION_PER_SHARE * 2
                equity    += pnl
                gross_win += max(pnl, 0)
                gross_loss += abs(min(pnl, 0))
                total_r   += r_mult
                if pnl > 0: wins += 1
                else: losses += 1
                peak   = max(peak, equity)
                dd     = (peak - equity) / peak * 100 if peak > 0 else 0
                max_dd = max(max_dd, dd)
                in_position = False
            continue

        if not bool(row.get("after_buffer", False)): continue
        bias = row.get("bias")
        if bias not in ("long", "short"): continue
        if not near_ema(row, config.ema_confluence_pct): continue
        if pd.isna(row.get("atr")): continue
        pattern = detect_pattern(sf, i, bias, config.enabled_patterns)
        if not pattern: continue
        if i + 1 >= len(sf): continue

        next_row     = sf.iloc[i + 1]
        signal_close = float(row["close"])
        raw_entry    = float(next_row["open"])
        entry_price  = raw_entry * (1 + SLIPPAGE_PCT) if bias == "long" else raw_entry * (1 - SLIPPAGE_PCT)

        atr_buffer    = float(row["atr"]) * float(config.atr_stop_buffer_mult)
        min_buffer    = max(0.05, signal_close * 0.0005)
        eff_buf       = max(atr_buffer, min_buffer)

        if bias == "long":
            stop_price   = float(row["low"])  - eff_buf
            target_price = entry_price + (entry_price - stop_price) * float(config.rr_target)
        else:
            stop_price   = float(row["high"]) + eff_buf
            target_price = entry_price - (stop_price - entry_price) * float(config.rr_target)

        risk = abs(entry_price - stop_price)
        if risk <= 0 or stop_price <= 0 or target_price <= 0: continue

        safe_risk   = max(risk, MIN_RISK_PER_SHARE)
        qty_risk    = int((equity * RISK_PCT) // safe_risk)
        qty_cap     = int((equity * MAX_CAPITAL_PCT) // entry_price)
        qty         = max(1, min(qty_risk, qty_cap))

        in_position = True
        side        = bias
        entry       = entry_price
        stop        = stop_price
        target      = target_price
        trade_qty   = qty

    total_trades = wins + losses
    if total_trades < 10:
        return None
    pf = gross_win / gross_loss if gross_loss > 0 else 999.0
    return {
        "total_trades":  total_trades,
        "wins":          wins,
        "losses":        losses,
        "win_rate_pct":  round(wins / total_trades * 100, 2),
        "profit_factor": round(pf, 3),
        "avg_rr":        round(total_r / total_trades, 3),
        "total_r":       round(total_r, 2),
        "max_dd_pct":    round(max_dd, 2),
        "net_pnl":       round(equity - capital, 2),
        "return_pct":    round((equity - capital) / capital * 100, 2),
        "final_equity":  round(equity, 2),
    }


_WORKER_DF = None

def _init_worker(df):
    """Called once per worker process at pool startup - loads df into memory ONCE."""
    global _WORKER_DF
    _WORKER_DF = df

def _run_combo(packed):
    params, capital, symbol = packed
    cfg = StrategyConfig()
    cfg.symbol = symbol
    cfg.initial_capital = capital
    for k, v in params.items():
        setattr(cfg, k, v)
    try:
        stats = run_backtest(_WORKER_DF, cfg)
    except Exception:
        return None
    if stats is None:
        return None
    row = {**params, **stats}
    row["enabled_patterns"] = "+".join(params["enabled_patterns"])
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol",  default="QQQ")
    parser.add_argument("--days",    type=int,   default=365)
    parser.add_argument("--capital", type=float, default=10000.0)
    parser.add_argument("--top",     type=int,   default=20)
    parser.add_argument("--workers", type=int,   default=6,
                        help="Parallel processes (default 6; tune to your CPU count)")
    parser.add_argument("--sample", type=float, default=1.0,
                        help="Fraction of combos to randomly test, e.g. 0.1 for 10%% fast pass")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible sampling")
    args = parser.parse_args()

    print(f"Fetching {args.days}d of 5-min bars for {args.symbol}...")
    df = fetch_bars(args.symbol, args.days)
    if df.empty:
        print("No data fetched — check API keys / symbol."); return
    print(f"Got {len(df)} bars.")

    combos = [dict(zip(PARAM_GRID.keys(), c)) for c in itertools.product(*PARAM_GRID.values())]
    if args.sample < 1.0:
        import random
        random.seed(args.seed)
        n_sample = max(1, int(len(combos) * args.sample))
        combos = random.sample(combos, n_sample)
        print(f"Sampling {args.sample*100:.1f}% of full grid -> {len(combos)} combos (seed={args.seed})")
    total  = len(combos)
    print(f"Running {total} combinations with {args.workers} parallel workers...")
    print(f"Costs: slippage={SLIPPAGE_PCT*100:.3f}%/fill  commission=${COMMISSION_PER_SHARE}/share  "
          f"risk={RISK_PCT*100:.1f}%  cap={MAX_CAPITAL_PCT*100:.0f}%\n")

    job_args = [(c, args.capital, args.symbol) for c in combos]
    results  = []
    done     = 0

    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(df,)) as pool:
        futures = {pool.submit(_run_combo, a): a for a in job_args}
        for fut in as_completed(futures):
            done += 1
            if done % 1000 == 0:
                print(f"  {done}/{total} done...")
            r = fut.result()
            if r is not None:
                results.append(r)

    if not results:
        print("No valid combos found (all had <10 trades or errored)."); return

    rdf = pd.DataFrame(results)
    rdf["breakeven_wr"] = (1 / (1 + rdf["rr_target"]) * 100).round(2)
    rdf["edge_above_be"] = (rdf["win_rate_pct"] - rdf["breakeven_wr"]).round(2)
    rdf = rdf.sort_values(["edge_above_be", "profit_factor", "return_pct"], ascending=False).reset_index(drop=True)
    rdf.to_csv("sweep_results.csv", index=False)
    print(f"Saved {len(rdf)} valid combos to sweep_results.csv\n")

    cols = [
        "entry_ema_length", "trend_sma_length", "ema_confluence_pct",
        "atr_length", "atr_stop_buffer_mult", "rr_target",
        "no_trade_buffer_minutes", "enabled_patterns",
        "total_trades", "win_rate_pct", "breakeven_wr", "edge_above_be",
        "profit_factor", "max_dd_pct", "return_pct", "net_pnl",
    ]
    print(f"TOP {args.top} COMBOS (sorted by edge above breakeven win rate):")
    print("=" * 170)
    print(rdf.head(args.top)[cols].to_string(index=False))

    best = rdf.iloc[0]
    print(f"""
{'='*60}
BEST COMBO SUMMARY
{'='*60}
  entry_ema_length       : {int(best['entry_ema_length'])}
  trend_sma_length       : {int(best['trend_sma_length'])}
  ema_confluence_pct     : {best['ema_confluence_pct']}
  atr_length             : {int(best['atr_length'])}
  atr_stop_buffer_mult   : {best['atr_stop_buffer_mult']}
  rr_target              : {best['rr_target']}
  no_trade_buffer_mins   : {int(best['no_trade_buffer_minutes'])}
  enabled_patterns       : {best['enabled_patterns']}
  ---
  total_trades           : {int(best['total_trades'])}
  win_rate_pct           : {best['win_rate_pct']}%
  breakeven_wr           : {best['breakeven_wr']}%
  edge_above_be          : +{best['edge_above_be']}%
  profit_factor          : {best['profit_factor']}
  max_dd_pct             : {best['max_dd_pct']}%
  return_pct             : {best['return_pct']}%
  net_pnl                : ${best['net_pnl']}
""")


if __name__ == "__main__":
    main()
