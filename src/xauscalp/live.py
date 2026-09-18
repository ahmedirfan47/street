"""Live / paper runner.

Safety posture, in order of precedence:

  * `execution.mode: paper` is the default. In paper mode no order ever reaches
    the broker, but positions ARE tracked, stops ARE resolved against each
    completed bar, and P/L IS recorded. Market data is genuine - real spreads,
    real gaps, real timing - so what you are testing is the strategy, not a
    simulation of the market.
  * Live mode on a REAL (non-demo) account additionally requires
    `allow_live_on_real_account: true`. Two independent switches, because one is
    too easy to flip by accident.
  * The loop acts only on CLOSED bars. Acting on a forming bar produces signals
    that disappear on the next tick.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

from .broker.base import Broker, OrderResult, Position
from .broker.mt5_broker import MT5Broker
from .config import Config
from .engine import BarContext, Engine
from .journal.store import Journal
from .risk.sizing import SymbolSpec

log = logging.getLogger(__name__)
UTC = timezone.utc


@dataclass
class PaperTrade:
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

    def as_journal_row(self) -> dict:
        return {
            "ticket": self.ticket,
            "opened_at": self.opened_at.isoformat(),
            "closed_at": self.closed_at.isoformat(),
            "session": self.session,
            "setup": self.setup,
            "direction": self.direction,
            "volume": self.volume,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "stop": self.meta.get("stop"),
            "pnl": self.pnl,
            "commission": self.commission,
            "r_multiple": self.r_multiple,
            "exit_reason": self.reason,
            "profile": self.meta.get("profile"),
        }


class PaperBroker:
    """Real market data in, simulated fills out.

    Reads (ticks, bars, symbol spec) pass through to the live MT5 connection, so
    spreads, gaps and data timing are genuine. Writes are simulated: nothing
    reaches the broker, but positions ARE tracked, stops ARE resolved against
    each completed bar, and P/L IS recorded - which is the entire point of
    forward testing.
    """

    def __init__(self, inner: Broker, cfg: Config, spec: SymbolSpec,
                 start_equity: float):
        self._inner = inner
        self.cfg = cfg
        self.spec = spec
        self.start_equity = start_equity
        self.balance = start_equity
        self._positions: dict[int, Position] = {}
        self._next_ticket = 900000
        self.closed: list[PaperTrade] = []
        self._last_price: float | None = None
        self._commission_paid = 0.0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    # ------------------------------------------------------------------ state
    def positions(self, symbol=None) -> list[Position]:
        return list(self._positions.values())

    def equity(self) -> float:
        eq = self.balance
        price = self._last_price
        if price is None:
            return eq
        vpu = self.spec.value_per_price_unit()
        for p in self._positions.values():
            move = (price - p.entry_price) if p.direction == "long" \
                else (p.entry_price - price)
            eq += move * vpu * p.volume
        return eq

    # ----------------------------------------------------------------- orders
    def market_order(self, symbol, direction, volume, stop, take_profit, comment=""):
        if volume <= 0:
            return OrderResult(False, error="zero_volume")
        tick = self._inner.tick(symbol)
        slip = self.spec.price(self.cfg.costs.assumed_slippage_points)
        # Fill at the far side plus assumed slippage - never better than mid.
        price = (tick.ask + slip) if direction == "long" else (tick.bid - slip)

        commission = self.cfg.costs.commission_per_lot_per_side * volume
        self.balance -= commission
        self._commission_paid += commission

        ticket = self._next_ticket
        self._next_ticket += 1
        pos = Position(
            ticket=ticket, symbol=symbol, direction=direction, volume=volume,
            entry_price=price, stop=stop, take_profit=take_profit,
            opened_at=self._inner.now(), initial_volume=volume, initial_stop=stop,
            comment=comment, magic=self.cfg.execution.magic,
        )
        self._positions[ticket] = pos
        self._last_price = tick.mid
        log.info("[PAPER] OPEN %s %.2f @%.2f SL %.2f TP %.2f (%s)",
                 direction, volume, price, stop, take_profit, comment)
        return OrderResult(True, ticket=ticket, price=price, volume=volume)

    def modify_position(self, ticket, stop=None, take_profit=None):
        pos = self._positions.get(ticket)
        if pos is None:
            return OrderResult(False, error="no_position")
        if stop is not None:
            pos.stop = stop
        if take_profit is not None:
            pos.take_profit = take_profit
        return OrderResult(True, ticket=ticket)

    def close_position(self, ticket, volume=None, reason="manual", price=None):
        pos = self._positions.get(ticket)
        if pos is None:
            return OrderResult(False, error="no_position")
        vol = pos.volume if volume is None else min(volume, pos.volume)
        vol = self.spec.normalize_volume(vol)
        if vol <= 0:
            return OrderResult(False, error="zero_volume")

        if price is None:
            tick = self._inner.tick(pos.symbol)
            slip = self.spec.price(self.cfg.costs.assumed_slippage_points)
            price = (tick.bid - slip) if pos.direction == "long" else (tick.ask + slip)
            self._last_price = tick.mid

        self._book(pos, vol, price, reason)
        return OrderResult(True, ticket=ticket, price=price, volume=vol)

    def _book(self, pos: Position, vol: float, price: float, reason: str) -> None:
        vpu = self.spec.value_per_price_unit()
        move = (price - pos.entry_price) if pos.direction == "long" \
            else (pos.entry_price - price)
        pnl = move * vpu * vol
        commission = self.cfg.costs.commission_per_lot_per_side * vol
        self.balance += pnl - commission
        self._commission_paid += commission

        risk = pos.initial_risk
        meta = dict(pos.meta)
        meta["stop"] = pos.initial_stop
        self.closed.append(PaperTrade(
            ticket=pos.ticket, direction=pos.direction, volume=vol,
            entry_price=pos.entry_price, exit_price=price,
            opened_at=pos.opened_at, closed_at=self._inner.now(),
            pnl=pnl - commission, commission=commission, reason=reason,
            r_multiple=(move / risk) if risk > 0 else 0.0,
            setup=pos.meta.get("setup", ""), session=pos.meta.get("session", ""),
            meta=meta,
        ))
        log.info("[PAPER] CLOSE %s %.2f @%.2f  pnl %+.2f  (%s)",
                 pos.direction, vol, price, pnl - commission, reason)

        # Same clamping trap as the simulator: compute the remainder BEFORE
        # normalising, or a fully closed position returns as volume_min and
        # gets re-booked forever.
        remaining = round(pos.volume - vol, 8)
        if remaining < self.spec.volume_min - 1e-9:
            self._positions.pop(pos.ticket, None)
        else:
            pos.volume = self.spec.normalize_volume(remaining)

    # ------------------------------------------------------------- bar resolve
    def resolve_bar(self, bar, ts: datetime) -> list[PaperTrade]:
        """Resolve SL/TP against a COMPLETED bar. Call once per new bar.

        Pessimistic, matching the backtester: a bar spanning both stop and
        target is assumed to have hit the stop first, and a gap through the
        stop fills at the bar open rather than the stop price.
        """
        before = len(self.closed)
        o = float(bar["open"]); h = float(bar["high"]); l = float(bar["low"])
        self._last_price = float(bar["close"])

        tick = self._inner.tick(self.cfg.symbol.name)
        half = (tick.ask - tick.bid) / 2.0
        slip = self.spec.price(self.cfg.costs.assumed_slippage_points)

        for pos in list(self._positions.values()):
            if pos.direction == "long":
                if pos.stop > 0 and (l - half) <= pos.stop:
                    fill = min(pos.stop, o - half) - slip
                    self._book(pos, pos.volume, fill, "stop_loss")
                    continue
                if pos.take_profit > 0 and (h - half) >= pos.take_profit:
                    self._book(pos, pos.volume, pos.take_profit, "take_profit")
            else:
                if pos.stop > 0 and (h + half) >= pos.stop:
                    fill = max(pos.stop, o + half) + slip
                    self._book(pos, pos.volume, fill, "stop_loss")
                    continue
                if pos.take_profit > 0 and (l + half) <= pos.take_profit:
                    self._book(pos, pos.volume, pos.take_profit, "take_profit")

        return self.closed[before:]

    # ---------------------------------------------------------------- reporting
    def summary(self) -> str:
        if not self.closed:
            return "  no closed paper trades yet"
        by_setup: dict[str, dict] = {}
        for t in self.closed:
            k = t.setup or "unknown"
            row = by_setup.setdefault(k, {"n": 0, "pnl": 0.0, "wins": 0})
            row["n"] += 1
            row["pnl"] += t.pnl
            row["wins"] += 1 if t.pnl > 0 else 0

        eq = self.equity()
        lines = [
            "",
            "-" * 66,
            f"  PAPER STATUS   equity {eq:,.2f}  "
            f"({(eq - self.start_equity) / self.start_equity * 100:+.2f}%)",
            f"  balance {self.balance:,.2f}   open {len(self._positions)}   "
            f"fills {len(self.closed)}",
            "-" * 66,
        ]
        for k, r in sorted(by_setup.items(), key=lambda x: -x[1]["pnl"]):
            wr = r["wins"] / r["n"] * 100 if r["n"] else 0.0
            lines.append(f"    {k:<18} n={r['n']:<4} pnl={r['pnl']:+9.2f}  wr={wr:5.1f}%")
        lines.append("-" * 66)
        return "\n".join(lines)


def run_live(cfg: Config, login=None, password=None, server=None,
             terminal_path=None) -> None:
    mt5b = MT5Broker(cfg, login, password, server, terminal_path)
    if not mt5b.connect():
        raise SystemExit("Could not connect to MT5. Is the terminal running and "
                         "'Algo Trading' enabled?")

    acct = mt5b.account()
    mode = cfg.execution.mode.lower()
    spec = mt5b.symbol_spec(cfg.symbol.name)

    log.info("Symbol spec: point=%s digits=%s contract=%s vol[%s..%s step %s] "
             "tick_value=%s tick_size=%s value_per_price_unit=%.2f",
             spec.point, spec.digits, spec.contract_size, spec.volume_min,
             spec.volume_max, spec.volume_step, spec.tick_value, spec.tick_size,
             spec.value_per_price_unit())

    paper: PaperBroker | None = None
    if mode == "live":
        if not acct.is_demo and not cfg.execution.allow_live_on_real_account:
            raise SystemExit(
                "REFUSING to trade live on a real-money account.\n"
                "Set execution.allow_live_on_real_account: true in the config if "
                "this is genuinely what you intend. Forward-test on demo first."
            )
        broker = mt5b
        start_equity = acct.equity
        log.warning("LIVE MODE - real orders will be sent. Account %s (demo=%s)",
                    acct.login, acct.is_demo)
    else:
        # Paper equity starts from the backtest figure so results are directly
        # comparable to the backtests, not from a demo balance that may be huge.
        start_equity = cfg.backtest.initial_equity
        paper = PaperBroker(mt5b, cfg, spec, start_equity)
        broker = paper
        log.info("PAPER MODE - no orders sent. Simulated equity starts at %,.2f"
                 .replace(",", ""), start_equity)

    journal = Journal(cfg.db_path)
    engine = Engine(cfg, broker, spec, journal, start_equity)

    if engine.news.is_stale(datetime.now(UTC)):
        log.warning("News calendar has no future events. Update %s before trading "
                    "through data releases.", cfg.news.csv_path)

    last_bar_time: pd.Timestamp | None = None
    bars_seen = 0
    log.info("Engine running. Polling every %.1fs on closed %s bars. Ctrl-C to stop.",
             cfg.execution.poll_seconds, cfg.execution.bar_timeframe)

    try:
        while True:
            try:
                m5 = mt5b.bars(cfg.symbol.name, cfg.execution.bar_timeframe,
                               cfg.execution.history_bars)
                if m5.empty:
                    time.sleep(cfg.execution.poll_seconds)
                    continue

                bar_time = m5.index[-1]
                if last_bar_time is not None and bar_time <= last_bar_time:
                    time.sleep(cfg.execution.poll_seconds)
                    continue
                last_bar_time = bar_time
                bars_seen += 1

                # Resolve paper stops against the bar that just completed,
                # BEFORE the engine looks for new entries on it.
                if paper is not None:
                    for t in paper.resolve_bar(m5.iloc[-1], bar_time.to_pydatetime()):
                        journal.log_trade(t.as_journal_row())

                htf = mt5b.bars(cfg.symbol.name, cfg.execution.htf_timeframe, 500)
                m5e, htfe = engine.prepare_frames(m5, htf)

                acct = mt5b.account()
                equity = paper.equity() if paper else acct.equity
                balance = paper.balance if paper else acct.balance

                ctx = BarContext(
                    now=datetime.now(UTC),
                    m5=m5e, htf=htfe,
                    spread_points=mt5b.spread_points(cfg.symbol.name),
                    equity=equity, balance=balance,
                    positions=broker.positions(cfg.symbol.name),
                )
                engine.on_bar(ctx)
                journal.log_equity(ctx.now, balance, equity, len(ctx.positions))

                # Status every 12 bars (1 hour on M5).
                if paper is not None and bars_seen % 12 == 0:
                    log.info(paper.summary())

            except KeyboardInterrupt:
                raise
            except Exception:
                log.exception("Loop error - continuing")
                time.sleep(5.0)

            time.sleep(cfg.execution.poll_seconds)

    except KeyboardInterrupt:
        log.info("Shutdown requested.")
    finally:
        if paper is not None:
            log.info(paper.summary())
        journal.close()
        mt5b.disconnect()
        log.info("Disconnected. Engine stats: %s", engine.stats)