"""Key price levels - the places where resting orders actually sit.

Ranked by how much stop liquidity typically accumulates there. Prior-day and
Asian-range extremes dominate because they are the reference points the widest
set of participants use.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

from ..sessions import SessionClock, Session


@dataclass(frozen=True)
class Level:
    name: str
    price: float
    kind: str        # "high" | "low"
    weight: float    # 0..1, relative liquidity expectation
    formed_at: datetime | None = None


def _slice(df: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    return df.loc[(df.index >= start) & (df.index < end)]


def build_levels(
    df: pd.DataFrame,
    clock: SessionClock,
    now: datetime,
    swing_highs: list[float] | None = None,
    swing_lows: list[float] | None = None,
) -> list[Level]:
    """Assemble the level map used by the sweep engine.

    `df` must be timezone-aware UTC indexed.
    """
    levels: list[Level] = []

    # ---- Asian range (the classic London-open target) ---------------------------
    ar = clock.asian_range_bounds(now)
    asia = _slice(df, ar.start_utc, ar.end_utc)
    if not asia.empty:
        levels.append(Level("asian_high", float(asia["high"].max()), "high", 1.00, ar.end_utc))
        levels.append(Level("asian_low", float(asia["low"].min()), "low", 1.00, ar.end_utc))

    # ---- Prior day high/low/close ----------------------------------------------
    day_anchor = (now + timedelta(hours=2)).date()
    prev = day_anchor - timedelta(days=1)
    while prev.weekday() >= 5:
        prev -= timedelta(days=1)
    pd_start = datetime.combine(prev, datetime.min.time(), tzinfo=now.tzinfo) - timedelta(hours=2)
    pd_end = pd_start + timedelta(days=1)
    prior = _slice(df, pd_start, pd_end)
    if not prior.empty:
        levels.append(Level("pdh", float(prior["high"].max()), "high", 0.95, pd_end))
        levels.append(Level("pdl", float(prior["low"].min()), "low", 0.95, pd_end))

    # ---- Prior session extremes -------------------------------------------------
    for session, w in ((Session.LONDON, 0.80), (Session.NEWYORK, 0.80)):
        for b in clock.bounds_around(now, session):
            if b.end_utc >= now:
                continue
            seg = _slice(df, b.start_utc, b.end_utc)
            if seg.empty:
                continue
            levels.append(Level(f"{session.value}_high", float(seg["high"].max()),
                                "high", w, b.end_utc))
            levels.append(Level(f"{session.value}_low", float(seg["low"].min()),
                                "low", w, b.end_utc))

    # ---- Equal highs / lows (stop clusters) -------------------------------------
    if swing_highs:
        for p in _cluster(swing_highs, tol=_tick_tol(df)):
            levels.append(Level("equal_highs", p, "high", 0.70, None))
    if swing_lows:
        for p in _cluster(swing_lows, tol=_tick_tol(df)):
            levels.append(Level("equal_lows", p, "low", 0.70, None))

    # Deduplicate levels that are effectively the same price; keep the heavier one.
    return _dedupe(levels, tol=_tick_tol(df) * 2)


def _tick_tol(df: pd.DataFrame) -> float:
    """Tolerance scaled to recent range - fixed dollar values break when gold
    reprices from $2,000 to $4,400."""
    if df.empty:
        return 0.10
    rng = float((df["high"] - df["low"]).tail(200).median())
    return max(0.05, rng * 0.12)


def _cluster(prices: list[float], tol: float) -> list[float]:
    """Return the mean of each group of >=2 near-identical prices."""
    if not prices:
        return []
    ordered = sorted(prices)
    groups: list[list[float]] = [[ordered[0]]]
    for p in ordered[1:]:
        if abs(p - groups[-1][-1]) <= tol:
            groups[-1].append(p)
        else:
            groups.append([p])
    return [sum(g) / len(g) for g in groups if len(g) >= 2]


def _dedupe(levels: list[Level], tol: float) -> list[Level]:
    out: list[Level] = []
    for lv in sorted(levels, key=lambda x: (-x.weight, x.price)):
        if any(abs(lv.price - o.price) <= tol and lv.kind == o.kind for o in out):
            continue
        out.append(lv)
    return sorted(out, key=lambda x: x.price)


def opposing_liquidity(levels: list[Level], price: float, direction: str) -> Level | None:
    """The nearest meaningful level the trade can realistically run to.

    If there is nothing to aim at, the setup has no natural target and gets
    rejected - a target invented from an R-multiple alone is wishful thinking.
    """
    if direction == "long":
        candidates = [l for l in levels if l.price > price and l.kind == "high"]
        return min(candidates, key=lambda l: l.price) if candidates else None
    candidates = [l for l in levels if l.price < price and l.kind == "low"]
    return max(candidates, key=lambda l: l.price) if candidates else None