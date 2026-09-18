"""Setups E and F - the day-trading workhorses.

The four original setups all wait for a specific structural event: a sweep, an
imbalance, a Fibonacci leg. Those are scarce on M15 - roughly one every two
days. The two here fire on a *schedule* instead of on rare structure, which is
what actually gets a day-trading system to a useful number of trades.

    E. Opening Range Breakout  - one setup per session, every session.
    F. VWAP Trend Pullback     - repeats through any trending session.

Both are directional by construction, so they sit naturally inside the session
direction lock: an ORB only takes the break on the locked side, and a VWAP
pullback only buys dips in an up-locked session.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from ..analytics.levels import Level, opposing_liquidity
from ..analytics.structure import find_swings, last_swing
from ..config import Config
from ..sessions import Session, SessionBounds
from .base import Rejection, SessionPlan, Signal


# ======================================================================
# Setup E - Opening Range Breakout
# ======================================================================
@dataclass(frozen=True)
class OpeningRange:
    high: float
    low: float
    start_index: int
    end_index: int

    @property
    def size(self) -> float:
        return self.high - self.low

    @property
    def mid(self) -> float:
        return (self.high + self.low) / 2.0


def build_opening_range(df: pd.DataFrame, bounds: SessionBounds,
                        orb_bars: int) -> OpeningRange | None:
    """High/low of the first `orb_bars` bars after the session opened.

    Returns None until the range is complete - trading a half-formed opening
    range is just trading noise with extra steps.
    """
    mask = np.asarray(df.index >= bounds.start_utc)
    idx = np.flatnonzero(mask)
    if len(idx) < orb_bars + 1:
        return None
    seg = df.iloc[idx[0]: idx[0] + orb_bars]
    if len(seg) < orb_bars:
        return None
    return OpeningRange(
        high=float(seg["high"].max()), low=float(seg["low"].min()),
        start_index=int(idx[0]), end_index=int(idx[0] + orb_bars - 1),
    )


def find_orb_signal(
    cfg: Config,
    df: pd.DataFrame,
    plan: SessionPlan,
    session: Session,
    bounds: SessionBounds | None,
    levels: list[Level],
    now: datetime,
    used_keys: set[str],
    min_viable_stop: float = 0.0,
) -> tuple[Signal | None, list[Rejection]]:
    """Break of the session opening range, in the session's locked direction."""
    rejections: list[Rejection] = []
    c = cfg.orb

    if "orb" not in plan.allowed_setups or bounds is None or len(df) < 40:
        return None, rejections

    atr_v = float(df["atr"].iloc[-1])
    if atr_v <= 0 or pd.isna(atr_v):
        return None, rejections

    orb = build_opening_range(df, bounds, c.orb_bars)
    if orb is None:
        return None, rejections

    last = len(df) - 1
    if last <= orb.end_index:
        return None, rejections

    # A range that is tiny or enormous relative to ATR is not a usable reference.
    if orb.size < c.min_range_atr * atr_v:
        rejections.append(Rejection(now, session, "orb", "range_too_tight",
                                    {"size": round(orb.size, 2), "atr": round(atr_v, 2)}))
        return None, rejections
    if orb.size > c.max_range_atr * atr_v:
        rejections.append(Rejection(now, session, "orb", "range_too_wide",
                                    {"size": round(orb.size, 2)}))
        return None, rejections

    bars_since = last - orb.end_index
    if bars_since > c.max_bars_after_open:
        return None, rejections

    close = float(df["close"].iloc[last])
    key = f"orb:{session.value}:{bounds.start_utc.date()}:{orb.high:.2f}"
    if key in used_keys:
        return None, rejections

    buf = c.break_buffer_atr * atr_v
    if close > orb.high + buf:
        direction = "long"
    elif close < orb.low - buf:
        direction = "short"
    else:
        return None, rejections

    # The lock is the whole point: only break in the direction the session
    # already committed to. A break the other way is a failed breakout.
    if plan.direction in ("long", "short") and direction != plan.direction:
        rejections.append(Rejection(now, session, "orb", "break_against_lock",
                                    {"break": direction, "lock": plan.direction}))
        return None, rejections

    entry = close
    # Stop at the far side of the range, capped so a wide range cannot produce
    # an unusable stop.
    raw_stop = orb.low - buf if direction == "long" else orb.high + buf
    stop_dist = abs(entry - raw_stop)
    max_stop = c.max_stop_atr * atr_v
    if stop_dist > max_stop:
        stop_dist = max_stop
    floor = max(c.min_stop_atr * atr_v, min_viable_stop)
    if stop_dist < floor:
        stop_dist = floor
    stop = entry - stop_dist if direction == "long" else entry + stop_dist

    # Targets projected from the range itself, which is the classic ORB measure.
    tp1 = entry + (c.tp1_range_mult * orb.size if direction == "long"
                   else -c.tp1_range_mult * orb.size)
    tp2 = entry + (c.tp2_range_mult * orb.size if direction == "long"
                   else -c.tp2_range_mult * orb.size)

    target_level = opposing_liquidity(levels, entry, direction)
    if target_level is not None:
        pad = 0.15 * atr_v
        lvl = target_level.price - pad if direction == "long" else target_level.price + pad
        capped = min(tp2, lvl) if direction == "long" else max(tp2, lvl)
        if abs(capped - entry) >= abs(tp1 - entry):
            tp2 = capped

    if abs(tp1 - entry) < stop_dist * 0.8:
        rejections.append(Rejection(now, session, "orb", "target_too_close", {}))
        return None, rejections

    # Earlier breaks are better; a break six bars into the session means more
    # than one that happens near the close.
    quality = 0.45
    quality += 0.25 * max(0.0, 1.0 - bars_since / max(c.max_bars_after_open, 1))
    quality += 0.20 * min(1.0, plan.regime.score / 100.0)
    quality += 0.10 * min(1.0, plan.direction_conviction / 0.6)

    sig = Signal(
        ts=now, session=session, setup="orb", direction=direction,
        entry=entry, stop=stop, tp1=tp1, tp2=tp2,
        reason=(f"broke {session.value} opening range "
                f"[{orb.low:.2f}-{orb.high:.2f}] (${orb.size:.2f}) "
                f"{bars_since} bars after open"),
        quality=min(1.0, quality), level_name="orb",
        meta={"orb_high": orb.high, "orb_low": orb.low,
              "orb_size": round(orb.size, 2), "bars_since_open": bars_since,
              "atr": atr_v, "orb_key": key,
              "target_level": target_level.name if target_level else None},
    )
    return sig, rejections


# ======================================================================
# Setup F - VWAP Trend Pullback
# ======================================================================
def session_vwap_series(df: pd.DataFrame, bounds: SessionBounds) -> pd.Series | None:
    """VWAP anchored to this session's open, volume-weighted by tick volume."""
    seg = df.loc[df.index >= bounds.start_utc]
    if len(seg) < 3:
        return None
    tp = (seg["high"] + seg["low"] + seg["close"]) / 3.0
    vol = seg["tick_volume"] if "tick_volume" in seg.columns else pd.Series(1.0, index=seg.index)
    vol = vol.replace(0, 1.0).astype(float)
    return (tp * vol).cumsum() / vol.cumsum()


def find_vwap_pullback_signal(
    cfg: Config,
    df: pd.DataFrame,
    plan: SessionPlan,
    session: Session,
    bounds: SessionBounds | None,
    levels: list[Level],
    now: datetime,
    last_entry_bar: dict[str, int],
    bar_index: int,
    min_viable_stop: float = 0.0,
) -> tuple[Signal | None, list[Rejection]]:
    """Buy the dip to session VWAP in an up-locked session (and vice versa).

    VWAP is where the session's average participant is filled, so it acts as a
    magnet and a decision point. In a trending session, pullbacks into it are
    where trend followers add - which is why this repeats through the session
    rather than firing once.
    """
    rejections: list[Rejection] = []
    c = cfg.vwap

    if "vwap" not in plan.allowed_setups or bounds is None or len(df) < 40:
        return None, rejections

    direction = plan.direction
    if direction not in ("long", "short"):
        return None, rejections     # needs a committed side to pull back within

    atr_v = float(df["atr"].iloc[-1])
    if atr_v <= 0 or pd.isna(atr_v):
        return None, rejections

    vwap = session_vwap_series(df, bounds)
    if vwap is None or len(vwap) < c.min_bars_into_session:
        return None, rejections

    last = len(df) - 1
    cooldown = last_entry_bar.get("vwap")
    if cooldown is not None and bar_index - cooldown < c.cooldown_bars:
        return None, rejections

    v_now = float(vwap.iloc[-1])
    close = float(df["close"].iloc[last])
    low = float(df["low"].iloc[last])
    high = float(df["high"].iloc[last])
    open_ = float(df["open"].iloc[last])

    band = c.touch_band_atr * atr_v

    if direction == "long":
        touched = low <= v_now + band
        reclaimed = close > v_now and close > open_
        extended_ok = (close - v_now) <= c.max_extension_atr * atr_v
        # Price must be above VWAP overall - buying below it in an "up" session
        # means the session thesis is already failing.
        thesis_intact = close >= v_now - band
    else:
        touched = high >= v_now - band
        reclaimed = close < v_now and close < open_
        extended_ok = (v_now - close) <= c.max_extension_atr * atr_v
        thesis_intact = close <= v_now + band

    if not touched:
        return None, rejections
    if not thesis_intact:
        rejections.append(Rejection(now, session, "vwap", "wrong_side_of_vwap",
                                    {"close": round(close, 2), "vwap": round(v_now, 2)}))
        return None, rejections
    if not reclaimed:
        rejections.append(Rejection(now, session, "vwap", "awaiting_reclaim", {}))
        return None, rejections
    if not extended_ok:
        rejections.append(Rejection(now, session, "vwap", "too_extended", {}))
        return None, rejections

    entry = close

    # Stop beyond the pullback swing, or beyond VWAP by a buffer.
    swings = find_swings(df.tail(60), left=2, right=2)
    buf = c.sl_buffer_atr * atr_v
    if direction == "long":
        anchor = last_swing(swings, "low", before_index=len(df.tail(60)) - 1)
        base = min(anchor.price, v_now) if anchor else v_now
        stop = base - buf
    else:
        anchor = last_swing(swings, "high", before_index=len(df.tail(60)) - 1)
        base = max(anchor.price, v_now) if anchor else v_now
        stop = base + buf

    stop_dist = abs(entry - stop)
    floor = max(c.min_stop_atr * atr_v, min_viable_stop)
    if stop_dist < floor:
        stop_dist = floor
        stop = entry - floor if direction == "long" else entry + floor
    if stop_dist > c.max_stop_atr * atr_v:
        rejections.append(Rejection(now, session, "vwap", "stop_too_wide",
                                    {"stop_dist": round(stop_dist, 2)}))
        return None, rejections

    tp1 = entry + (c.tp1_r * stop_dist if direction == "long" else -c.tp1_r * stop_dist)
    tp2 = entry + (c.tp2_r * stop_dist if direction == "long" else -c.tp2_r * stop_dist)

    target_level = opposing_liquidity(levels, entry, direction)
    if target_level is not None:
        pad = 0.15 * atr_v
        lvl = target_level.price - pad if direction == "long" else target_level.price + pad
        capped = min(tp2, lvl) if direction == "long" else max(tp2, lvl)
        if abs(capped - entry) >= abs(tp1 - entry):
            tp2 = capped

    dist_atr = abs(close - v_now) / atr_v
    quality = 0.40
    quality += 0.25 * max(0.0, 1.0 - dist_atr / max(c.max_extension_atr, 0.1))
    quality += 0.20 * min(1.0, plan.regime.score / 100.0)
    quality += 0.15 * min(1.0, plan.direction_conviction / 0.6)

    sig = Signal(
        ts=now, session=session, setup="vwap", direction=direction,
        entry=entry, stop=stop, tp1=tp1, tp2=tp2,
        reason=(f"pullback to session VWAP {v_now:.2f} reclaimed "
                f"({dist_atr:.2f} ATR away)"),
        quality=min(1.0, quality), level_name="vwap",
        meta={"vwap": round(v_now, 2), "dist_atr": round(dist_atr, 3),
              "atr": atr_v,
              "target_level": target_level.name if target_level else None},
    )
    return sig, rejections