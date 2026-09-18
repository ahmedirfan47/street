"""Synthetic XAUUSD bar generator.

Used ONLY for testing the machinery when MT5 is unavailable. It reproduces the
structural features the strategy depends on - session-dependent volatility, an
Asian compression phase, a London expansion, mean-reverting overshoots at range
extremes - so that the code paths get exercised.

It is NOT a substitute for real data. Any performance number produced on
synthetic bars tells you the code runs, not that the strategy works. Backtest
on real MT5 history before drawing any conclusion.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

UTC = timezone.utc


def _session_vol_multiplier(hour_utc: int) -> float:
    """Approximate intraday volatility seasonality for gold."""
    if 0 <= hour_utc < 6:        # Asian
        return 0.55
    if 6 <= hour_utc < 7:        # pre-London
        return 0.80
    if 7 <= hour_utc < 12:       # London
        return 1.35
    if 12 <= hour_utc < 16:      # London/NY overlap
        return 1.55
    if 16 <= hour_utc < 20:      # NY afternoon
        return 0.95
    return 0.45                  # late NY / rollover


def generate_bars(
    start: str = "2025-06-01",
    end: str = "2026-08-01",
    timeframe_minutes: int = 5,
    start_price: float = 4400.0,
    seed: int = 42,
    base_bar_vol: float = 1.45,
) -> pd.DataFrame:
    """Generate OHLCV bars with realistic gold-like microstructure."""
    rng = np.random.default_rng(seed)

    start_dt = datetime.fromisoformat(start).replace(tzinfo=UTC)
    end_dt = datetime.fromisoformat(end).replace(tzinfo=UTC)

    stamps: list[datetime] = []
    t = start_dt
    step = timedelta(minutes=timeframe_minutes)
    while t < end_dt:
        # Market closed: Friday 21:00 UTC through Sunday 22:00 UTC.
        wd = t.weekday()
        closed = (wd == 5) or (wd == 4 and t.hour >= 21) or (wd == 6 and t.hour < 22)
        if not closed:
            stamps.append(t)
        t += step

    n = len(stamps)
    if n == 0:
        raise ValueError("No bars generated - check the date range")

    # Slow-moving volatility regime (GARCH-ish persistence).
    regime = np.zeros(n)
    regime[0] = 1.0
    for i in range(1, n):
        regime[i] = 0.9985 * regime[i - 1] + 0.0015 * rng.lognormal(0.0, 0.55)
    regime = np.clip(regime, 0.35, 3.2)

    # Multi-day drift component so trends and ranges both appear.
    drift = np.zeros(n)
    d = 0.0
    for i in range(n):
        d = 0.9992 * d + rng.normal(0.0, 0.010)
        drift[i] = d

    closes = np.zeros(n)
    highs = np.zeros(n)
    lows = np.zeros(n)
    opens = np.zeros(n)
    vols = np.zeros(n)

    price = start_price
    session_anchor = price
    anchor_hour = -1

    for i, ts in enumerate(stamps):
        sv = _session_vol_multiplier(ts.hour)
        sigma = base_bar_vol * sv * regime[i]

        # Weak mean reversion toward a slowly-updating session anchor. This is
        # what creates the overshoot-and-reclaim behaviour at range extremes.
        if ts.hour != anchor_hour:
            anchor_hour = ts.hour
            session_anchor = 0.985 * session_anchor + 0.015 * price
        pull = -0.012 * (price - session_anchor)

        ret = rng.normal(drift[i] * 0.5, sigma) + pull

        # Occasional impulse (data print, headline).
        if rng.random() < 0.0016:
            ret += rng.normal(0.0, 1.0) * sigma * rng.uniform(5.0, 14.0)

        o = price
        c = price + ret
        wick = abs(rng.normal(0.0, sigma * 0.85))
        h = max(o, c) + wick * rng.uniform(0.25, 1.0)
        l = min(o, c) - wick * rng.uniform(0.25, 1.0)

        opens[i], closes[i], highs[i], lows[i] = o, c, h, l
        vols[i] = max(1.0, rng.normal(400 * sv, 120 * sv))
        price = c

    df = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes,
         "tick_volume": vols},
        index=pd.DatetimeIndex(stamps, name="time"),
    )
    # Guarantee OHLC consistency.
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    return df


def resample(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Aggregate to a higher timeframe."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "tick_volume" in df.columns:
        agg["tick_volume"] = "sum"
    out = df.resample(f"{minutes}min").agg(agg).dropna(subset=["open"])
    return out