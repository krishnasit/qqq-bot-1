from dataclasses import dataclass, field
from typing import List


@dataclass
class StrategyConfig:
    symbol: str = 'QQQ'
    timeframe_minutes: int = 5
    london_timezone: str = 'Europe/London'
    bias_start_hour: int = 8
    bias_start_minute: int = 0
    no_trade_buffer_minutes: int = 90
    trend_sma_length: int = 200
    entry_ema_length: int = 50
    ema_confluence_pct: float = 0.8
    atr_length: int = 20
    atr_stop_buffer_mult: float = 0.25
    rr_target: float = 1.5
    trailing_stop_enabled: bool = False
    trailing_atr_mult: float = 1.0
    enabled_patterns: List[str] = field(default_factory=lambda: ['engulfing'])
    initial_capital: float = 10000.0
