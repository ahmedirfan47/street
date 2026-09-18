"""Simulated broker for backtesting and paper trading.

Fill assumptions are deliberately pessimistic:

  * Market orders fill at the far side of the spread plus slippage.
  * Stops fill at the stop price plus slippage *against* you; when a bar gaps
    through the stop, the fill is at the bar open, not the stop price. Optimistic
    stop fills are the single most common reason a backtest looks profitable and
    live trading does not.
  * If a bar's range spans both the stop and the target, the stop is assumed to
    have been hit first, unless intrabar data says otherwise.
  * Commission is charged on both sides.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from ..config import Config
from ..risk.sizing import SymbolSpec
from .base import AccountInfo, Broker, OrderResult, Position, Tick


@dataclass
class ClosedTrade:
    ticket: int
    direction: str
    volume: float
    entry_price: float
    exit_price: float
    opened_at: datetime
    closed_at: datetime
    pnl: float
    commission: float
    reason: str
    r_multiple: float
    setup: str = ""
    session: str = ""
    meta: dict = field(default_factory=dict)


class SimBroker(Broker):
    def __init__(self, cfg: Config, bars: pd.DataFrame, spec: SymbolSpec,
                 seed: int = 7):
        self.cfg = cfg
        self._all_bars = bars
        self._spec = spec
        self._i = 0
        self._balance = cfg.backtest.initial_equity
        self._positions: dict[int, Position] = {}
        self._next_ticket = 1000
        self.closed: list[ClosedTrade] = []
        self.equity_curve: list[tuple[datetime, float]] = []
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self._spread_points = cfg.backtest.spread_points_mean
        self._commission_paid = 0.0

    # ---------------------------------------------------------------- plumbing
    def connect(self) -> bool:
        return True

    def disconnect(self) -> None:
        return None

    def symbol_spec(self, symbol: str) -> SymbolSpec:
        return self._spec

    def now(self) -> datetime:
        return self._all_bars.index[self._i].to_pydatetime()

    @property
    def current_bar(self) -> pd.Series:
        return self._all_bars.iloc[self._i]

    def account(self) -> AccountInfo:
        return AccountInfo(balance=self._balance, equity=self.equity(),
                           is_demo=True, server="SIM")

    def equity(self) -> float:
        eq = self._balance
        price = float(self.current_bar["close"])
        vpu = self._spec.value_per_price_unit()
        for p in self._positions.values():
            move = (price - p.entry_price) if p.direction == "long" \
                else (p.entry_price - price)
            eq += move * vpu * p.volume
        return eq

    # ------------------------------------------------------------------ market
    def _sample_spread(self) -> float:
        b = self.cfg.backtest
        s = self._np_rng.normal(b.spread_points_mean, b.spread_points_std)
        # Spreads are bounded below and fat-tailed above.
        s = max(b.spread_points_mean * 0.35, s)
        if self._rng.random() < 0.02:      # occasional liquidity air pocket
            s *= self._rng.uniform(2.0, 4.5)
        self._spread_points = s
        return s

    def tick(self, symbol: str) -> Tick:
        bar = self.current_bar
        mid = float(bar["close"])
        half = self._spec.price(self._spread_points) / 2.0
        return Tick(self.now(), mid - half, mid + half)

    def spread_points(self) -> float:
        return self._spread_points

    def bars(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        lo = max(0, self._i - count + 1)
        return self._all_bars.iloc[lo: self._i + 1].copy()

    def positions(self, symbol: str | None = None) -> list[Position]:
        return list(self._positions.values())

    # ------------------------------------------------------------------ orders
    def market_order(self, symbol: str, direction: str, volume: float,
                     stop: float, take_profit: float,
                     comment: str = "") -> OrderResult:
        if volume <= 0:
            return OrderResult(False, error="zero_volume")

        bar = self.current_bar
        mid = float(bar["close"])
        half = self._spec.price(self._spread_points) / 2.0
        slip = self._spec.price(
            max(0.0, self._np_rng.normal(self.cfg.backtest.slippage_points_mean,
                                         self.cfg.backtest.slippage_points_mean * 0.6))
        )
        fill = mid + half + slip if direction == "long" else mid - half - slip

        commission = self.cfg.costs.commission_per_lot_per_side * volume
        self._balance -= commission
        self._commission_paid += commission

        ticket = self._next_ticket
        self._next_ticket += 1
        pos = Position(
            ticket=ticket, symbol=symbol, direction=direction, volume=volume,
            entry_price=fill, stop=stop, take_profit=take_profit,
            opened_at=self.now(), initial_volume=volume, initial_stop=stop,
            comment=comment, magic=self.cfg.execution.magic,
        )
        self._positions[ticket] = pos
        return OrderResult(True, ticket=ticket, price=fill, volume=volume)

    def modify_position(self, ticket: int, stop: float | None = None,
                        take_profit: float | None = None) -> OrderResult:
        pos = self._positions.get(ticket)
        if pos is None:
            return OrderResult(False, error="no_position")
        if stop is not None:
            pos.stop = stop
        if take_profit is not None:
            pos.take_profit = take_profit
        return OrderResult(True, ticket=ticket)

    def close_position(self, ticket: int, volume: float | None = None,
                       reason: str = "manual", price: float | None = None) -> OrderResult:
        pos = self._positions.get(ticket)
        if pos is None:
            return OrderResult(False, error="no_position")

        vol = pos.volume if volume is None else min(volume, pos.volume)
        vol = self._spec.normalize_volume(vol)
        if vol <= 0:
            return OrderResult(False, error="zero_volume")

        bar = self.current_bar
        if price is None:
            mid = float(bar["close"])
            half = self._spec.price(self._spread_points) / 2.0
            slip = self._spec.price(
                max(0.0, self._np_rng.normal(self.cfg.backtest.slippage_points_mean,
                                             self.cfg.backtest.slippage_points_mean * 0.6))
            )
            price = mid - half - slip if pos.direction == "long" else mid + half + slip

        self._book(pos, vol, price, reason)
        return OrderResult(True, ticket=ticket, price=price, volume=vol)

    def _book(self, pos: Position, vol: float, price: float, reason: str) -> None:
        vpu = self._spec.value_per_price_unit()
        move = (price - pos.entry_price) if pos.direction == "long" \
            else (pos.entry_price - price)
        pnl = move * vpu * vol
        commission = self.cfg.costs.commission_per_lot_per_side * vol
        self._balance += pnl - commission
        self._commission_paid += commission

        risk = pos.initial_risk
        r = (move / risk) if risk > 0 else 0.0

        self.closed.append(ClosedTrade(
            ticket=pos.ticket, direction=pos.direction, volume=vol,
            entry_price=pos.entry_price, exit_price=price,
            opened_at=pos.opened_at, closed_at=self.now(),
            pnl=pnl - commission, commission=commission, reason=reason,
            r_multiple=r,
            setup=pos.meta.get("setup", ""), session=pos.meta.get("session", ""),
            meta=dict(pos.meta),
        ))

        # CRITICAL: compute the remainder BEFORE normalising. normalize_volume()
        # clamps UP to volume_min, so a fully-closed position would come back as
        # 0.01 lots, survive the removal check, and be re-booked on every
        # subsequent bar - an unbounded loss spiral. Compare the raw remainder.
        remaining = round(pos.volume - vol, 8)
        if remaining < self._spec.volume_min - 1e-9:
            self._positions.pop(pos.ticket, None)
        else:
            pos.volume = self._spec.normalize_volume(remaining)

    # ------------------------------------------------------------- bar stepping
    def step(self) -> bool:
        """Advance one bar, resolving SL/TP against the new bar's range."""
        if self._i >= len(self._all_bars) - 1:
            return False
        self._i += 1
        self._sample_spread()
        self._resolve_stops()
        self.equity_curve.append((self.now(), self.equity()))
        return True

    def _resolve_stops(self) -> None:
        bar = self.current_bar
        o, h, l = float(bar["open"]), float(bar["high"]), float(bar["low"])
        half = self._spec.price(self._spread_points) / 2.0
        slip = self._spec.price(self.cfg.backtest.slippage_points_mean)

        for pos in list(self._positions.values()):
            if pos.direction == "long":
                # Bid is what closes a long, so compare against bid = mid - half.
                stop_hit = (l - half) <= pos.stop
                tp_hit = pos.take_profit > 0 and (h - half) >= pos.take_profit
                if stop_hit:
                    # Gap-through: fill at open, not at the stop price.
                    fill = min(pos.stop, o - half) - slip
                    self._book(pos, pos.volume, fill, "stop_loss")
                    continue
                if tp_hit:
                    fill = max(pos.take_profit, o - half) if o - half > pos.take_profit \
                        else pos.take_profit
                    self._book(pos, pos.volume, fill, "take_profit")
            else:
                stop_hit = (h + half) >= pos.stop
                tp_hit = pos.take_profit > 0 and (l + half) <= pos.take_profit
                if stop_hit:
                    fill = max(pos.stop, o + half) + slip
                    self._book(pos, pos.volume, fill, "stop_loss")
                    continue
                if tp_hit:
                    fill = min(pos.take_profit, o + half) if o + half < pos.take_profit \
                        else pos.take_profit
                    self._book(pos, pos.volume, fill, "take_profit")

    def seek(self, index: int) -> None:
        self._i = max(0, min(index, len(self._all_bars) - 1))

    @property
    def index(self) -> int:
        return self._i

    @property
    def total_commission(self) -> float:
        return self._commission_paid