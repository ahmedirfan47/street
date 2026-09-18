"""Session clock.

All internal timestamps are timezone-aware UTC. Session windows are defined in
the *local* time of the relevant financial centre and converted through zoneinfo,
so DST transitions in London and New York are handled without manual offset edits.
This matters: a hardcoded 07:00 UTC London open is wrong for half the year.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from enum import Enum
from zoneinfo import ZoneInfo

from .config import SessionsCfg, SessionWindow

UTC = timezone.utc


class Session(str, Enum):
    ASIAN = "asian"
    LONDON = "london"
    NEWYORK = "newyork"
    OFF = "off"


def _parse_hhmm(s: str) -> time:
    hh, mm = s.split(":")
    return time(int(hh), int(mm))


@dataclass(frozen=True)
class SessionBounds:
    session: Session
    start_utc: datetime
    end_utc: datetime

    def contains(self, ts: datetime) -> bool:
        return self.start_utc <= ts < self.end_utc

    @property
    def duration_minutes(self) -> float:
        return (self.end_utc - self.start_utc).total_seconds() / 60.0


class SessionClock:
    def __init__(self, cfg: SessionsCfg):
        self.cfg = cfg

    # ---------------------------------------------------------------- windows
    def _bounds_for_local_date(
        self, window: SessionWindow, local_date, session: Session
    ) -> SessionBounds:
        tz = ZoneInfo(window.tz)
        start_local = datetime.combine(local_date, _parse_hhmm(window.start), tzinfo=tz)
        end_local = datetime.combine(local_date, _parse_hhmm(window.end), tzinfo=tz)
        if end_local <= start_local:  # window crosses local midnight
            end_local += timedelta(days=1)
        return SessionBounds(session, start_local.astimezone(UTC), end_local.astimezone(UTC))

    def bounds_around(self, ts_utc: datetime, session: Session) -> list[SessionBounds]:
        """Candidate session windows near ts (yesterday/today/tomorrow local)."""
        window = getattr(self.cfg, session.value)
        tz = ZoneInfo(window.tz)
        local_today = ts_utc.astimezone(tz).date()
        out = []
        for delta in (-1, 0, 1):
            d = local_today + timedelta(days=delta)
            if d.weekday() >= 5:  # skip Sat/Sun local
                continue
            out.append(self._bounds_for_local_date(window, d, session))
        return out

    def active_session(self, ts_utc: datetime) -> Session:
        """Priority: London > NewYork > Asian when windows overlap.

        London/NY overlap is the highest-liquidity regime; treating it as London
        keeps continuation logic on the session that established the move.
        """
        for session in (Session.LONDON, Session.NEWYORK, Session.ASIAN):
            window = getattr(self.cfg, session.value)
            if not window.enabled:
                continue
            for b in self.bounds_around(ts_utc, session):
                if b.contains(ts_utc):
                    return session
        return Session.OFF

    def current_bounds(self, ts_utc: datetime, session: Session) -> SessionBounds | None:
        for b in self.bounds_around(ts_utc, session):
            if b.contains(ts_utc):
                return b
        return None

    def next_open(self, ts_utc: datetime) -> tuple[Session, datetime] | None:
        candidates: list[tuple[datetime, Session]] = []
        for session in (Session.ASIAN, Session.LONDON, Session.NEWYORK):
            window = getattr(self.cfg, session.value)
            if not window.enabled:
                continue
            for b in self.bounds_around(ts_utc, session):
                if b.start_utc > ts_utc:
                    candidates.append((b.start_utc, session))
        if not candidates:
            return None
        candidates.sort()
        return candidates[0][1], candidates[0][0]

    def is_plan_time(self, ts_utc: datetime) -> tuple[Session, datetime] | None:
        """True inside the pre-session analysis lead window."""
        nxt = self.next_open(ts_utc)
        if nxt is None:
            return None
        session, open_utc = nxt
        lead = timedelta(minutes=self.cfg.plan_lead_minutes)
        if open_utc - lead <= ts_utc < open_utc:
            return session, open_utc
        return None

    def minutes_to_session_end(self, ts_utc: datetime, session: Session) -> float:
        b = self.current_bounds(ts_utc, session)
        if b is None:
            return 0.0
        return (b.end_utc - ts_utc).total_seconds() / 60.0

    def asian_range_bounds(self, ts_utc: datetime) -> SessionBounds:
        """The Asian accumulation window whose high/low London will target."""
        window = SessionWindow(
            tz=self.cfg.asian.tz,
            start=self.cfg.asian_range_start,
            end=self.cfg.asian_range_end,
        )
        tz = ZoneInfo(window.tz)
        local_now = ts_utc.astimezone(tz)
        d = local_now.date()
        b = self._bounds_for_local_date(window, d, Session.ASIAN)
        # If today's Asian range has not finished yet, use yesterday's completed one.
        if ts_utc < b.end_utc:
            d = d - timedelta(days=1)
            while d.weekday() >= 5:
                d -= timedelta(days=1)
            b = self._bounds_for_local_date(window, d, Session.ASIAN)
        return b

    def is_friday_flatten(self, ts_utc: datetime) -> bool:
        if ts_utc.weekday() != 4:
            return False
        t = _parse_hhmm(self.cfg.friday_flatten_utc)
        return ts_utc.timetz().replace(tzinfo=None) >= t

    def trading_day_key(self, ts_utc: datetime) -> str:
        """Trading day rolls at 22:00 UTC, matching the broker daily bar convention."""
        anchor = ts_utc + timedelta(hours=2)
        return anchor.strftime("%Y-%m-%d")