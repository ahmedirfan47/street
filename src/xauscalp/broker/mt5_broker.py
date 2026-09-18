"""Live MetaTrader 5 adapter.

The MetaTrader5 package is Windows-only and is imported lazily so the rest of
the system (backtester, tests, analysis) runs anywhere.

Two details that break most Python/MT5 bots and are handled explicitly here:

  1. Broker server time is not UTC. Most retail servers run UTC+2/+3, and it
     shifts with European DST. Bar timestamps come back in *server* time. This
     adapter measures the offset AND VALIDATES IT - an unchecked offset from a
     stale tick silently corrupts every session boundary while the bot keeps
     running and keeps printing plans.

  2. Filling modes differ per broker and symbol. Submitting the wrong one gets
     you 10030 "Unsupported filling mode". This adapter reads the symbol's
     supported modes and picks a valid one.
"""
from __future__ import annotations

import logging
import time as _time
from datetime import datetime, timedelta, timezone

import pandas as pd

from ..config import Config
from ..risk.sizing import SymbolSpec
from .base import AccountInfo, Broker, OrderResult, Position, Tick

log = logging.getLogger(__name__)
UTC = timezone.utc

_TF_MAP = {
    "M1": "TIMEFRAME_M1", "M2": "TIMEFRAME_M2", "M3": "TIMEFRAME_M3",
    "M5": "TIMEFRAME_M5", "M10": "TIMEFRAME_M10", "M15": "TIMEFRAME_M15",
    "M30": "TIMEFRAME_M30", "H1": "TIMEFRAME_H1", "H4": "TIMEFRAME_H4",
    "D1": "TIMEFRAME_D1",
}


class MT5Broker(Broker):
    # Realistic bounds for a forex broker's server offset. Almost all sit at
    # UTC+2 or UTC+3; nothing legitimate falls outside this window. Anything
    # else means the timestamp we measured was stale, not that the broker
    # moved to a different continent.
    MIN_OFFSET_H = -2
    MAX_OFFSET_H = 6

    def __init__(self, cfg: Config, login: int | None = None,
                 password: str | None = None, server: str | None = None,
                 terminal_path: str | None = None):
        self.cfg = cfg
        self._login = login
        self._password = password
        self._server = server
        self._terminal_path = terminal_path
        self._mt5 = None
        self._offset = timedelta(0)   # server_time - utc
        self._spec: SymbolSpec | None = None
        self._filling = None
        self._offset_suspect = False

    # ---------------------------------------------------------------- lifecycle
    @property
    def mt5(self):
        if self._mt5 is None:
            try:
                import MetaTrader5 as mt5  # noqa: N813
            except ImportError as exc:
                raise RuntimeError(
                    "MetaTrader5 package not available. It is Windows-only:\n"
                    "  pip install MetaTrader5\n"
                    "Run the backtester instead on other platforms."
                ) from exc
            self._mt5 = mt5
        return self._mt5

    def connect(self) -> bool:
        mt5 = self.mt5
        kwargs = {}
        if self._terminal_path:
            kwargs["path"] = self._terminal_path
        if self._login:
            kwargs.update(login=int(self._login), password=self._password,
                          server=self._server)
        if not mt5.initialize(**kwargs):
            log.error("MT5 initialize failed: %s", mt5.last_error())
            return False

        symbol = self.cfg.symbol.name
        if not mt5.symbol_select(symbol, True):
            log.error("Could not select symbol %s: %s", symbol, mt5.last_error())
            return False

        self._detect_time_offset(symbol)
        self._spec = self._read_spec(symbol)
        self._filling = self._pick_filling_mode(symbol)

        acct = self.account()
        log.info("MT5 connected | login=%s server=%s demo=%s equity=%.2f %s",
                 acct.login, acct.server, acct.is_demo, acct.equity, acct.currency)
        log.info("Server time offset vs UTC: %+.1f h%s",
                 self._offset.total_seconds() / 3600,
                 "  [SUSPECT - see error above]" if self._offset_suspect else "")
        return True

    def disconnect(self) -> None:
        if self._mt5 is not None:
            self._mt5.shutdown()

    # ------------------------------------------------------------------- clocks
    def _latest_server_timestamp(self, symbol: str) -> datetime | None:
        """Most recent server-time stamp available, from ticks OR bars.

        A tick can be hours stale when the terminal has just started, when the
        symbol is not actively subscribed, or when the feed hiccups. The latest
        M1 bar is a second, independent source; taking whichever is more recent
        makes the measurement far harder to poison.
        """
        candidates: list[datetime] = []

        # Retry the tick a few times - a fresh one usually arrives within a
        # second or two once the symbol is properly subscribed.
        for _ in range(4):
            t = self.mt5.symbol_info_tick(symbol)
            if t is not None and t.time:
                candidates.append(datetime.fromtimestamp(t.time, tz=UTC))
                break
            _time.sleep(0.4)

        # Latest completed M1 bar, as a cross-check.
        try:
            rates = self.mt5.copy_rates_from_pos(symbol, self.mt5.TIMEFRAME_M1, 0, 1)
            if rates is not None and len(rates) > 0:
                candidates.append(
                    datetime.fromtimestamp(int(rates[-1]["time"]), tz=UTC)
                )
        except Exception as exc:
            log.debug("M1 cross-check failed: %s", exc)

        return max(candidates) if candidates else None

    def _detect_time_offset(self, symbol: str) -> None:
        """Server time minus UTC.

        Validated, not merely measured. An unchecked offset silently corrupts
        every session boundary in the system - the bot keeps running and keeps
        printing plans, it just believes London opens during the Asian session.
        """
        override = self.cfg.execution.server_utc_offset_hours
        if override is not None:
            self._offset = timedelta(hours=float(override))
            log.info("Server offset FIXED by config at %+.1f h", float(override))
            return

        server_dt = self._latest_server_timestamp(symbol)
        if server_dt is None:
            log.error("No tick or bar for %s. Cannot measure server offset; "
                      "assuming UTC. Set execution.server_utc_offset_hours "
                      "in the config if you know it.", symbol)
            self._offset = timedelta(0)
            return

        real_utc = datetime.now(tz=UTC)
        raw = (server_dt - real_utc).total_seconds() / 3600.0
        snapped = round(raw)

        if not (self.MIN_OFFSET_H <= snapped <= self.MAX_OFFSET_H):
            log.error(
                "REFUSING implausible server offset %+.1f h (raw %+.2f h).\n"
                "  Latest server stamp : %s\n"
                "  Real UTC now        : %s\n"
                "This means the timestamp was STALE, not that the broker uses "
                "that offset. Every session boundary would be wrong.\n"
                "  -> Is the market open? Is XAUUSD in Market Watch and ticking?\n"
                "  -> If you know the offset, set execution.server_utc_offset_hours.\n"
                "Falling back to +3 (the common UTC+3 broker default).",
                snapped, raw, server_dt.isoformat(), real_utc.isoformat(),
            )
            self._offset = timedelta(hours=3)
            self._offset_suspect = True
            return

        # Warn on a stale-but-plausible stamp: outside market hours the last
        # tick is legitimately old, and the offset is then only approximate.
        staleness_min = abs(raw - snapped) * 60.0
        if staleness_min > 20.0:
            log.warning("Server stamp looks %.0f min off a whole hour - the feed "
                        "may be stale. Offset %+d h used, verify when the market "
                        "is active.", staleness_min, snapped)

        self._offset = timedelta(hours=snapped)
        self._offset_suspect = False

    def server_to_utc(self, ts) -> datetime:
        if isinstance(ts, (int, float)):
            ts = datetime.fromtimestamp(ts, tz=UTC)
        elif ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        return ts - self._offset

    def utc_to_server(self, ts: datetime) -> datetime:
        return ts + self._offset

    def now(self) -> datetime:
        return datetime.now(tz=UTC)

    # -------------------------------------------------------------------- specs
    def _read_spec(self, symbol: str) -> SymbolSpec:
        info = self.mt5.symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"symbol_info({symbol}) returned None")
        return SymbolSpec(
            name=symbol,
            point=info.point,
            digits=info.digits,
            contract_size=info.trade_contract_size,
            volume_min=info.volume_min,
            volume_max=info.volume_max,
            volume_step=info.volume_step,
            tick_value=info.trade_tick_value,
            tick_size=info.trade_tick_size,
            stops_level_points=getattr(info, "trade_stops_level", 0) or 0,
        )

    def symbol_spec(self, symbol: str) -> SymbolSpec:
        if self._spec is None:
            self._spec = self._read_spec(symbol)
        return self._spec

    def _pick_filling_mode(self, symbol: str):
        mt5 = self.mt5
        info = mt5.symbol_info(symbol)
        mask = getattr(info, "filling_mode", 0)
        # SYMBOL_FILLING_FOK = 1, SYMBOL_FILLING_IOC = 2
        if mask & 2:
            return mt5.ORDER_FILLING_IOC
        if mask & 1:
            return mt5.ORDER_FILLING_FOK
        return mt5.ORDER_FILLING_RETURN

    # ------------------------------------------------------------------ account
    def account(self) -> AccountInfo:
        a = self.mt5.account_info()
        if a is None:
            raise RuntimeError("account_info() returned None")
        is_demo = getattr(a, "trade_mode", 0) == getattr(
            self.mt5, "ACCOUNT_TRADE_MODE_DEMO", 0
        )
        return AccountInfo(balance=a.balance, equity=a.equity, currency=a.currency,
                           leverage=a.leverage, is_demo=is_demo, login=a.login,
                           server=a.server)

    # -------------------------------------------------------------- market data
    def tick(self, symbol: str) -> Tick:
        t = self.mt5.symbol_info_tick(symbol)
        if t is None:
            raise RuntimeError(f"symbol_info_tick({symbol}) returned None")
        return Tick(self.server_to_utc(t.time), t.bid, t.ask)

    def spread_points(self, symbol: str) -> float:
        spec = self.symbol_spec(symbol)
        t = self.tick(symbol)
        return (t.ask - t.bid) / spec.point if spec.point > 0 else 0.0

    def bars(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        mt5 = self.mt5
        tf_name = _TF_MAP.get(timeframe.upper())
        if tf_name is None:
            raise ValueError(f"Unsupported timeframe {timeframe}")
        tf = getattr(mt5, tf_name)

        rates = None
        for attempt in range(self.cfg.execution.retry_attempts):
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
            if rates is not None and len(rates) > 0:
                break
            _time.sleep(self.cfg.execution.retry_sleep_seconds)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"No rates for {symbol} {timeframe}: {mt5.last_error()}")

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True) - self._offset
        df = df.set_index("time").sort_index()
        cols = ["open", "high", "low", "close"]
        if "tick_volume" in df.columns:
            cols.append("tick_volume")
        if "spread" in df.columns:
            cols.append("spread")
        df = df[cols].astype(float)
        # Drop the still-forming bar: acting on an incomplete candle is lookahead
        # in reverse and produces signals that vanish on the next tick.
        return df.iloc[:-1]

    # ---------------------------------------------------------------- positions
    def positions(self, symbol: str | None = None) -> list[Position]:
        sym = symbol or self.cfg.symbol.name
        raw = self.mt5.positions_get(symbol=sym)
        if raw is None:
            return []
        out: list[Position] = []
        for p in raw:
            if p.magic != self.cfg.execution.magic:
                continue
            direction = "long" if p.type == self.mt5.POSITION_TYPE_BUY else "short"
            out.append(Position(
                ticket=p.ticket, symbol=p.symbol, direction=direction,
                volume=p.volume, entry_price=p.price_open, stop=p.sl,
                take_profit=p.tp, opened_at=self.server_to_utc(p.time),
                initial_volume=p.volume, initial_stop=p.sl,
                comment=p.comment, magic=p.magic,
            ))
        return out

    # ------------------------------------------------------------------- orders
    def market_order(self, symbol: str, direction: str, volume: float,
                     stop: float, take_profit: float,
                     comment: str = "") -> OrderResult:
        mt5 = self.mt5
        spec = self.symbol_spec(symbol)
        t = self.tick(symbol)
        price = t.ask if direction == "long" else t.bid
        order_type = mt5.ORDER_TYPE_BUY if direction == "long" else mt5.ORDER_TYPE_SELL

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(spec.normalize_volume(volume)),
            "type": order_type,
            "price": float(price),
            "sl": float(spec.normalize_price(stop)) if stop else 0.0,
            "tp": float(spec.normalize_price(take_profit)) if take_profit else 0.0,
            "deviation": int(self.cfg.execution.deviation_points),
            "magic": int(self.cfg.execution.magic),
            "comment": (comment or self.cfg.execution.comment)[:31],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._filling,
        }

        for attempt in range(self.cfg.execution.retry_attempts):
            result = mt5.order_send(request)
            if result is None:
                log.error("order_send returned None: %s", mt5.last_error())
                _time.sleep(self.cfg.execution.retry_sleep_seconds)
                continue
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                return OrderResult(True, ticket=result.order, price=result.price,
                                   volume=result.volume, raw=result)
            if result.retcode in (mt5.TRADE_RETCODE_REQUOTE,
                                  mt5.TRADE_RETCODE_PRICE_OFF,
                                  mt5.TRADE_RETCODE_PRICE_CHANGED):
                t = self.tick(symbol)
                request["price"] = float(t.ask if direction == "long" else t.bid)
                _time.sleep(self.cfg.execution.retry_sleep_seconds)
                continue
            return OrderResult(False, error=f"retcode={result.retcode} {result.comment}",
                               raw=result)
        return OrderResult(False, error="order_send exhausted retries")

    def modify_position(self, ticket: int, stop: float | None = None,
                        take_profit: float | None = None) -> OrderResult:
        mt5 = self.mt5
        pos = [p for p in (mt5.positions_get(ticket=ticket) or [])]
        if not pos:
            return OrderResult(False, error="no_position")
        p = pos[0]
        spec = self.symbol_spec(p.symbol)
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": p.symbol,
            "position": int(ticket),
            "sl": float(spec.normalize_price(stop if stop is not None else p.sl)),
            "tp": float(spec.normalize_price(
                take_profit if take_profit is not None else p.tp)),
            "magic": int(self.cfg.execution.magic),
        }
        result = mt5.order_send(request)
        if result is None:
            return OrderResult(False, error=str(mt5.last_error()))
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            return OrderResult(True, ticket=ticket, raw=result)
        return OrderResult(False, error=f"retcode={result.retcode} {result.comment}",
                           raw=result)

    def close_position(self, ticket: int, volume: float | None = None) -> OrderResult:
        mt5 = self.mt5
        raw = mt5.positions_get(ticket=ticket)
        if not raw:
            return OrderResult(False, error="no_position")
        p = raw[0]
        spec = self.symbol_spec(p.symbol)
        vol = spec.normalize_volume(p.volume if volume is None else min(volume, p.volume))
        if vol <= 0:
            return OrderResult(False, error="zero_volume")

        t = self.tick(p.symbol)
        if p.type == mt5.POSITION_TYPE_BUY:
            order_type, price = mt5.ORDER_TYPE_SELL, t.bid
        else:
            order_type, price = mt5.ORDER_TYPE_BUY, t.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": p.symbol,
            "volume": float(vol),
            "type": order_type,
            "position": int(ticket),
            "price": float(price),
            "deviation": int(self.cfg.execution.deviation_points),
            "magic": int(self.cfg.execution.magic),
            "comment": "xauscalp_close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._filling,
        }
        for _ in range(self.cfg.execution.retry_attempts):
            result = mt5.order_send(request)
            if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                return OrderResult(True, ticket=ticket, price=result.price,
                                   volume=result.volume, raw=result)
            t = self.tick(p.symbol)
            request["price"] = float(t.bid if p.type == mt5.POSITION_TYPE_BUY else t.ask)
            _time.sleep(self.cfg.execution.retry_sleep_seconds)
        return OrderResult(False, error="close exhausted retries")