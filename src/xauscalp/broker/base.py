"""Broker abstraction.

MT5 and the simulator implement the same surface, which means the strategy,
risk and execution code is identical in backtest and live. Any behaviour that
only exists on one side is a backtest that lies to you.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from ..risk.sizing import SymbolSpec


@dataclass
class Tick:
    ts: datetime
    bid: float
    ask: float

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def mid(self) -> float:
        return (self.ask + self.bid) / 2.0


@dataclass
class Position:
    ticket: int
    symbol: str
    direction: str          # "long" | "short"
    volume: float
    entry_price: float
    stop: float
    take_profit: float
    opened_at: datetime
    initial_volume: float
    initial_stop: float
    comment: str = ""
    magic: int = 0
    partial_taken: bool = False
    moved_to_be: bool = False
    trailing: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def initial_risk(self) -> float:
        return abs(self.entry_price - self.initial_stop)

    def r_multiple(self, price: float) -> float:
        risk = self.initial_risk
        if risk <= 0:
            return 0.0
        move = (price - self.entry_price) if self.direction == "long" \
            else (self.entry_price - price)
        return move / risk


@dataclass
class OrderResult:
    ok: bool
    ticket: int | None = None
    price: float | None = None
    volume: float | None = None
    error: str = ""
    raw: Any = None


@dataclass
class AccountInfo:
    balance: float
    equity: float
    currency: str = "USD"
    leverage: int = 100
    is_demo: bool = True
    login: int = 0
    server: str = ""


class Broker(ABC):
    @abstractmethod
    def connect(self) -> bool: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def symbol_spec(self, symbol: str) -> SymbolSpec: ...

    @abstractmethod
    def account(self) -> AccountInfo: ...

    @abstractmethod
    def tick(self, symbol: str) -> Tick: ...

    @abstractmethod
    def bars(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame: ...

    @abstractmethod
    def positions(self, symbol: str | None = None) -> list[Position]: ...

    @abstractmethod
    def market_order(self, symbol: str, direction: str, volume: float,
                     stop: float, take_profit: float,
                     comment: str = "") -> OrderResult: ...

    @abstractmethod
    def modify_position(self, ticket: int, stop: float | None = None,
                        take_profit: float | None = None) -> OrderResult: ...

    @abstractmethod
    def close_position(self, ticket: int, volume: float | None = None) -> OrderResult: ...

    @abstractmethod
    def now(self) -> datetime: ...