<<<<<<< HEAD
# qqq-bot-1
=======
# QQQ London Bias Bot

A QQQ 5-minute London-bias strategy project with a Streamlit backtester and an Alpaca paper-trading worker.

## Final strategy defaults

- Symbol: QQQ
- Timeframe: 5 minutes
- London no-trade buffer: 90 minutes
- Trend SMA: 200
- Entry EMA: 50
- EMA confluence: 0.80%
- Enabled pattern: engulfing only
- ATR length: 20
- ATR stop buffer: 0.25x ATR
- Reward/risk target: 1.5R
- Trailing stop: disabled

## Local backtester

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

## Paper bot

`paper_bot.py` uses Alpaca paper trading only. It evaluates completed 5-minute candles, prevents duplicate positions and orders, and submits bracket orders with attached stop-loss and take-profit levels.

### Test one cycle

```bash
python3 paper_bot.py --once
```

### Run continuously

```bash
python3 paper_bot.py --loop
```

Orders submitted by the worker are logged to `paper_trades.csv`.

## Railway deployment

Railway starts:

```bash
python3 paper_bot.py --loop
```

Add these Railway Variables:

- `ALPACA_API_KEY`
- `ALPACA_SECRET_KEY`
- `ALPACA_PAPER=true`
- `BOT_SYMBOL=QQQ`
- `BOT_QTY=1`
- `BOT_POLL_SECONDS=60`
- `BOT_LOG_PATH=paper_trades.csv`

Do not commit `.env` or real credentials.
>>>>>>> 3bdccf1 (Prepare QQQ paper trading bot for Railway)
