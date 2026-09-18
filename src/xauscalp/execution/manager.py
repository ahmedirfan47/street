"""Open-position management.

Where a scalping system actually makes or loses its money. The entry decides
whether you have a trade; this decides what it is worth.

Sequence per position:
  1. At +1R (configurable): close a partial and move the stop to break-even
     plus a spread-sized offset, so the remainder is genuinely free.
  2. Past the trail trigger: chandelier trail at N x ATR from the running
     extreme, ratcheting only in the favourable direction.
  3. Time stop: a scalp that has not made progress within N minutes is closed.
     Gold intraday edges decay fast; holding a stale position converts a small
     cost into a full stop-out.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ..broker.base import Broker, Position
from ..config import Config
from ..risk.sizing import SymbolSpec

log = logging.getLogger(__name__)


@dataclass
class PositionState:
    ticket: int
    peak_price: float
    trough_price: float
    partial_taken: bool = False
    moved_to_be: bool = False
    trailing: bool = False
    opened_at: datetime | None = None
    meta: dict = field(default_factory=dict)


class TradeManager:
    def __init__(self, cfg: Config, broker: Broker, spec: SymbolSpec):
        self.cfg = cfg
        self.broker = broker
        self.spec = spec
        self.states: dict[int, PositionState] = {}

    def register(self, pos: Position, meta: dict | None = None) -> None:
        self.states[pos.ticket] = PositionState(
            ticket=pos.ticket, peak_price=pos.entry_price,
            trough_price=pos.entry_price, opened_at=pos.opened_at,
            meta=meta or {},
        )

    def forget(self, ticket: int) -> None:
        self.states.pop(ticket, None)

    def sync(self, positions: list[Position]) -> None:
        live = {p.ticket for p in positions}
        for t in list(self.states):
            if t not in live:
                self.forget(t)
        for p in positions:
            if p.ticket not in self.states:
                self.register(p)

    # ------------------------------------------------------------------ update
    def update(self, positions: list[Position], price: float, atr_value: float,
               now: datetime) -> list[str]:
        """Run the management rules. Returns a list of actions taken."""
        actions: list[str] = []
        self.sync(positions)
        m = self.cfg.manage

        for pos in positions:
            st = self.states.get(pos.ticket)
            if st is None:
                continue

            st.peak_price = max(st.peak_price, price)
            st.trough_price = min(st.trough_price, price)

            r = pos.r_multiple(price)
            risk = pos.initial_risk
            if risk <= 0:
                continue

            # ---- 1. partial + break-even ------------------------------------
            if not st.partial_taken and r >= m.move_to_be_at_r:
                if m.partial_at_tp1_pct > 0:
                    vol = self.spec.normalize_volume(
                        pos.volume * (m.partial_at_tp1_pct / 100.0)
                    )
                    if vol >= self.spec.volume_min and vol < pos.volume:
                        res = self.broker.close_position(pos.ticket, vol)
                        if res.ok:
                            actions.append(f"partial_close t{pos.ticket} {vol}")
                st.partial_taken = True

                offset = m.be_offset_atr * atr_value
                be_stop = (pos.entry_price + offset if pos.direction == "long"
                           else pos.entry_price - offset)
                improves = (be_stop > pos.stop if pos.direction == "long"
                            else be_stop < pos.stop)
                if improves:
                    res = self.broker.modify_position(pos.ticket, stop=be_stop)
                    if res.ok:
                        st.moved_to_be = True
                        actions.append(f"move_to_be t{pos.ticket} @{be_stop:.2f}")

            # ---- 2. chandelier trail ----------------------------------------
            if r >= m.trail_after_r and atr_value > 0:
                st.trailing = True
                dist = m.trail_atr_mult * atr_value
                new_stop = (st.peak_price - dist if pos.direction == "long"
                            else st.trough_price + dist)
                improves = (new_stop > pos.stop if pos.direction == "long"
                            else new_stop < pos.stop)
                # Never trail past the current price into an instant stop-out.
                safe = (new_stop < price if pos.direction == "long"
                        else new_stop > price)
                if improves and safe:
                    res = self.broker.modify_position(pos.ticket, stop=new_stop)
                    if res.ok:
                        actions.append(f"trail t{pos.ticket} @{new_stop:.2f}")

            # ---- 3. time stops ------------------------------------------------
            opened = st.opened_at or pos.opened_at
            if opened is not None:
                held = (now - opened).total_seconds() / 60.0
                if held >= m.max_hold_minutes:
                    res = self._close(pos, "max_hold")
                    if res:
                        actions.append(f"max_hold_close t{pos.ticket}")
                    continue
                if held >= m.time_stop_minutes and r < m.time_stop_min_r:
                    res = self._close(pos, "time_stop")
                    if res:
                        actions.append(f"time_stop_close t{pos.ticket} r={r:.2f}")
                    continue

        return actions

    def _close(self, pos: Position, reason: str) -> bool:
        try:
            res = self.broker.close_position(pos.ticket)
        except TypeError:
            res = self.broker.close_position(pos.ticket, None)
        if res.ok:
            self.forget(pos.ticket)
            log.info("Closed t%s (%s)", pos.ticket, reason)
            return True
        log.warning("Failed to close t%s (%s): %s", pos.ticket, reason, res.error)
        return False

    def flatten_all(self, positions: list[Position], reason: str) -> list[str]:
        out = []
        for p in positions:
            if self._close(p, reason):
                out.append(f"flatten t{p.ticket} ({reason})")
        return out