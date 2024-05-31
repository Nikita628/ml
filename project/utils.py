from datetime import datetime
from enum import Enum


def format_date(date: datetime) -> str:
    return date.strftime("%Y-%m-%d_%H-%M-%S")

class PredictionType(Enum):
    atr_mult = 'atr_mult'
    pct_increase = 'pct_increase'
    next_close_direction = 'next_close_direction'

NON_FEATURE_COLUMNS = ['timestamp', 'open', 'close', 'high', 'low', 'volume', 'file']