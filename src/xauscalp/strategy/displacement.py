"""Setup B - Displacement Continuation.

Hypothesis: an outsized directional candle that leaves an unfilled imbalance is
the footprint of size being worked. Price frequently retraces into that
imbalance before continuing, because the participants who missed the initial
move place resting bids/offers there.

This is the momentum-side complement to the sweep setup. It is only permitted
when the regime read is trending or rotational-with-direction, and it requires
alignment with the session bias - a continuation trade against the higher
timeframe is just a slower way to be wrong.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from ..analytics.levels import Level, opposing_liquidity
from ..analytics.structure import (
    FVG, find_fvgs, find_swings, fvg_is_fresh, is_displacement, last_swing,
)
from ..config import Config
from ..sessions import Session
from .base import Rejection, Signal, SessionPlan


def _quality(fvg: FVG, atr_v: float, plan: SessionPlan, age: int, cfg: Config) -> float:
    q = 0.0
    # Larger imbalance relative to ATR = stronger flow signature.
    q += 0.30 * min(1.0, (fvg.size / atr_v) / 0.6) if atr_v > 0 else 0.0
    # Fresher gaps are better.
    q += 0.25 * max(0.0, 1.0 - age / max(cfg.displacement.max_fvg_age_bars, 1))
    q += 0.25 * min(1.0, plan.regime.score / 100.0)
    if (plan.bias == "up" and fvg.direction == "bull") or \
       (plan.bias == "down" and fvg.direction == "bear"):
        q += 0.20
    elif plan.bias == "neutral":
        q += 0.08
    return float(min(1.0, max(0.0, q)))


def find_displacement_signal(
    cfg: Config,
    df: pd.DataFrame,
    plan: SessionPlan,
    session: Session,
    levels: list[Level],
    now: datetime,
    used_fvgs: set[str],
    min_viable_stop: float = 0.0,
) -> tuple[Signal | None, list[Rejection]]:
    """Fires when the just-closed bar trades into a fresh displacement FVG."""
    rejections: list[Rejection] = []
    dcfg = cfg.displacement

    if "displacement" not in plan.allowed_setups:
        return None, rejections
    if len(df) < 60:
        return None, rejections

    atr_v = float(df["atr"].iloc[-1])
    if atr_v <= 0 or pd.isna(atr_v):
        return None, rejections

    last = len(df) - 1
    min_gap = dcfg.min_fvg_atr * atr_v
    fvgs = find_fvgs(df, min_size=min_gap, lookback=dcfg.max_fvg_age_bars + 5)

    best: tuple[float, Signal] | None = None

    for fvg in fvgs:
        age = last - fvg.index
        if age < 1 or age > dcfg.max_fvg_age_bars:
            continue

        key = f"fvg:{fvg.direction}:{fvg.index}:{fvg.top:.2f}"
        if key in used_fvgs:
            continue

        # The displacement bar is the one that created the gap (bar index-1).
        disp = is_displacement(df, fvg.index - 1, atr_v, dcfg.body_atr_mult)
        if disp is None:
            disp = is_displacement(df, fvg.index, atr_v, dcfg.body_atr_mult)
        if disp is None or disp != fvg.direction:
            continue

        if not fvg_is_fresh(fvg, df.iloc[:last]):
            continue

        direction = "long" if fvg.direction == "bull" else "short"

        if dcfg.require_bias_alignment and plan.bias != "neutral":
            aligned = (plan.bias == "up" and direction == "long") or \
                      (plan.bias == "down" and direction == "short")
            if not aligned:
                rejections.append(Rejection(now, session, "displacement",
                                            "bias_misaligned",
                                            {"bias": plan.bias, "dir": direction}))
                continue

        # Trigger: the just-closed bar must have traded into the gap.
        bar_low = float(df["low"].iloc[last])
        bar_high = float(df["high"].iloc[last])
        entry_price = fvg.midpoint(dcfg.entry_fvg_fraction)

        if direction == "long":
            touched = bar_low <= fvg.top and bar_low >= fvg.bottom - 0.05 * atr_v
            invalidated = float(df["close"].iloc[last]) < fvg.bottom
        else:
            touched = bar_high >= fvg.bottom and bar_high <= fvg.top + 0.05 * atr_v
            invalidated = float(df["close"].iloc[last]) > fvg.top

        if not touched or invalidated:
            continue

        # Enter at the current close rather than assuming a limit fill at the
        # midpoint - the backtest and live path must agree on fill assumptions.
        entry = float(df["close"].iloc[last])

        # --- stop: beyond the origin of the displacement -----------------------
        swings = find_swings(df.tail(120), left=2, right=2)
        buf = dcfg.sl_buffer_atr * atr_v
        if direction == "long":
            anchor = last_swing(swings, "low", before_index=last)
            base = min(fvg.bottom, anchor.price) if anchor else fvg.bottom
            stop = base - buf
        else:
            anchor = last_swing(swings, "high", before_index=last)
            base = max(fvg.top, anchor.price) if anchor else fvg.top
            stop = base + buf

        stop_dist = abs(entry - stop)

        # Widen to the cost-viable minimum rather than being rejected later for
        # a friction ratio the setup itself had no say in.
        floor = max(dcfg.min_stop_atr * atr_v, min_viable_stop)
        if stop_dist < floor:
            stop_dist = floor
            stop = entry - floor if direction == "long" else entry + floor

        if stop_dist > dcfg.max_stop_atr * atr_v:
            rejections.append(Rejection(now, session, "displacement", "stop_too_wide",
                                        {"stop_dist": stop_dist, "atr": atr_v}))
            continue

        tp1 = entry + (dcfg.tp1_r * stop_dist if direction == "long"
                       else -dcfg.tp1_r * stop_dist)
        tp2 = entry + (dcfg.tp2_r * stop_dist if direction == "long"
                       else -dcfg.tp2_r * stop_dist)

        target_level = opposing_liquidity(levels, entry, direction)
        if target_level is not None:
            pad = 0.15 * atr_v
            lvl = target_level.price - pad if direction == "long" else target_level.price + pad
            # Do not aim past obvious liquidity; that is where continuation stalls.
            tp2 = min(tp2, lvl) if direction == "long" else max(tp2, lvl)
            if abs(tp2 - entry) < abs(tp1 - entry):
                rejections.append(Rejection(now, session, "displacement",
                                            "liquidity_too_close",
                                            {"target": target_level.name}))
                continue

        q = _quality(fvg, atr_v, plan, age, cfg)
        sig = Signal(
            ts=now, session=session, setup="displacement", direction=direction,
            entry=entry, stop=stop, tp1=tp1, tp2=tp2,
            reason=(f"{fvg.direction} displacement FVG "
                    f"[{fvg.bottom:.2f}-{fvg.top:.2f}] retraced after {age} bar(s)"),
            quality=q, level_name="fvg",
            meta={"fvg_top": fvg.top, "fvg_bottom": fvg.bottom, "fvg_age": age,
                  "atr": atr_v, "fvg_key": key,
                  "target_level": target_level.name if target_level else None},
        )
        if best is None or q > best[0]:
            best = (q, sig)

    return (best[1] if best else None), rejections