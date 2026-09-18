"""Economic calendar blackout.

Gold is a rates and dollar instrument. NFP, CPI, FOMC and Fed-chair remarks
routinely move it several dollars in seconds, with spreads widening 5-20x
through the print. No intraday structural edge survives that, so the bot simply
stands aside.

The calendar is a user-maintained CSV rather than a scraped feed, because free
calendar APIs are unreliable and a silently stale feed is worse than none. A
runtime spread-spike check in the guard stack acts as the backstop for anything
the CSV misses.

CSV format (UTC timestamps):
    datetime_utc,currency,impact,event
    2026-09-04 12:30,USD,high,Non-Farm Payrolls
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ..config import NewsCfg

log = logging.getLogger(__name__)
UTC = timezone.utc

# Currencies whose data actually moves XAUUSD.
RELEVANT = {"USD", "ALL", "XAU"}


@dataclass(frozen=True)
class Event:
    ts: datetime
    currency: str
    impact: str
    name: str


class NewsFilter:
    def __init__(self, cfg: NewsCfg):
        self.cfg = cfg
        self.events: list[Event] = []
        self._loaded_from: str | None = None
        if cfg.enabled:
            self.load(cfg.csv_path)

    def load(self, path: str | Path) -> int:
        p = Path(path)
        self.events = []
        if not p.exists():
            log.warning(
                "News calendar not found at %s - the bot will run WITHOUT event "
                "blackouts and rely only on the spread-spike guard. Populate it "
                "before trading around data.", p,
            )
            return 0
        blocked = {i.lower() for i in self.cfg.blocked_impacts}
        with p.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    raw = (row.get("datetime_utc") or "").strip()
                    if not raw:
                        continue
                    ts = datetime.fromisoformat(raw)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=UTC)
                    impact = (row.get("impact") or "").strip().lower()
                    currency = (row.get("currency") or "").strip().upper()
                    if impact not in blocked:
                        continue
                    if currency not in RELEVANT:
                        continue
                    self.events.append(
                        Event(ts.astimezone(UTC), currency, impact,
                              (row.get("event") or "").strip())
                    )
                except (ValueError, TypeError) as exc:
                    log.debug("Skipping malformed calendar row %s: %s", row, exc)
        self.events.sort(key=lambda e: e.ts)
        self._loaded_from = str(p)
        log.info("Loaded %d blocking events from %s", len(self.events), p)
        return len(self.events)

    def blackout(self, now: datetime) -> Event | None:
        """The event currently blacking out trading, if any."""
        if not self.cfg.enabled:
            return None
        before = timedelta(minutes=self.cfg.block_before_minutes)
        after = timedelta(minutes=self.cfg.block_after_minutes)
        for e in self.events:
            if e.ts - before <= now <= e.ts + after:
                return e
        return None

    def next_event(self, now: datetime) -> Event | None:
        for e in self.events:
            if e.ts > now:
                return e
        return None

    def minutes_to_next(self, now: datetime) -> float | None:
        e = self.next_event(now)
        if e is None:
            return None
        return (e.ts - now).total_seconds() / 60.0

    def is_stale(self, now: datetime) -> bool:
        """True if the calendar has no future events - i.e. nobody updated it."""
        if not self.cfg.enabled:
            return False
        return self.next_event(now) is None