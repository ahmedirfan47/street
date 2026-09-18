"""Setup C - Fibonacci Retracement Continuation.

Hypothesis: after a decisive impulse leg, price retraces to a proportional
fraction of that leg before continuing. Whether the retracement ratios carry
genuine information or simply mark levels enough participants watch to become
self-fulfilling is an open question - and it does not much matter, because the
trade is the same either way.

What matters is that this setup is *geometrically* attractive and *economically*
fragile. A deeper entry gives a tighter stop and a better R multiple, but the
tighter stop is also more likely to be clipped, and a stop that is too tight
cannot clear round-turn costs at all. The profile system exists so that trade-off
is settled with recorded results rather than conviction.

TWO PROFILES RUN IN PARALLEL. Every signal is tagged with the profile that
produced it, so the journal can compare them directly:

    SELECT json_extract(payload,'$.meta.profile') AS profile,
           COUNT(*), AVG(r_multiple), SUM(pnl)
    FROM trades WHERE setup LIKE 'fib%' GROUP BY 1;

Geometry, for a bullish leg from swing low L to swing high H (length D = H - L):

    entry zone  : H - D * entry_zone_start  ...  H - D * entry_zone_end
    stop        : H - D * stop_level, minus an ATR buffer
    tp1         : L + D * tp1_extension     (1.0 = back to the leg high)
    tp2         : L + D * tp2_extension     (1.618 = classic extension)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from ..analytics.levels import Level, opposing_liquidity
from ..analytics.structure import Swing, find_swings
from ..config import Config, FibProfile
from ..sessions import Session
from .base import Rejection, SessionPlan, Signal


@dataclass(frozen=True)
class ImpulseLeg:
    start_index: int
    end_index: int
    start_price: float
    end_price: float
    direction: str          # "long" = up leg, look for longs on the retrace

    @property
    def length(self) -> float:
        return abs(self.end_price - self.start_price)

    @property
    def bars(self) -> int:
        return self.end_index - self.start_index

    def retracement(self, fraction: float) -> float:
        """Price at `fraction` retraced back into the leg."""
        if self.direction == "long":
            return self.end_price - self.length * fraction
        return self.end_price + self.length * fraction

    def extension(self, fraction: float) -> float:
        """Price at `fraction` projected from the leg origin."""
        if self.direction == "long":
            return self.start_price + self.length * fraction
        return self.start_price - self.length * fraction

    def key(self) -> str:
        return f"leg:{self.direction}:{self.start_index}:{self.end_index}"


def find_impulse_legs(swings: list[Swing], profile: FibProfile,
                      atr_value: float, last_index: int) -> list[ImpulseLeg]:
    """Completed swing-to-swing legs that qualify as impulses.

    Uses confirmed swings only, so a leg cannot be recognised until `swing_right`
    bars after its extreme printed. That lag is the price of not reading the
    future.
    """
    legs: list[ImpulseLeg] = []
    if atr_value <= 0 or len(swings) < 2:
        return legs

    min_len = profile.min_leg_atr * atr_value
    max_len = profile.max_leg_atr * atr_value

    for i in range(1, len(swings)):
        a, b = swings[i - 1], swings[i]
        if a.kind == b.kind:
            continue  # need alternating high/low to form a leg

        direction = "long" if (a.kind == "low" and b.kind == "high") else "short"
        length = abs(b.price - a.price)
        bars = b.index - a.index

        if length < min_len or length > max_len:
            continue
        if bars < profile.min_leg_bars or bars > profile.max_leg_bars:
            continue
        if last_index - b.index > profile.max_leg_age_bars:
            continue

        legs.append(ImpulseLeg(a.index, b.index, a.price, b.price, direction))

    return legs


def _quality(leg: ImpulseLeg, profile: FibProfile, atr_value: float,
             plan: SessionPlan, depth: float, age: int) -> float:
    """0..1 confidence, before any cost consideration."""
    q = 0.0

    # Impulse strength relative to typical range.
    q += 0.25 * min(1.0, (leg.length / atr_value) / (profile.min_leg_atr * 2.0)) \
        if atr_value > 0 else 0.0

    # Steepness: a leg covering distance in fewer bars is more decisive.
    if leg.bars > 0 and atr_value > 0:
        per_bar = (leg.length / leg.bars) / atr_value
        q += 0.20 * min(1.0, per_bar / 0.8)

    # Retracement sitting mid-zone rather than at its edge.
    mid = (profile.entry_zone_start + profile.entry_zone_end) / 2.0
    half = max((profile.entry_zone_end - profile.entry_zone_start) / 2.0, 1e-6)
    q += 0.20 * max(0.0, 1.0 - abs(depth - mid) / half)

    # Freshness.
    q += 0.15 * max(0.0, 1.0 - age / max(profile.max_leg_age_bars, 1))

    # Regime.
    q += 0.10 * min(1.0, plan.regime.score / 100.0)

    # Bias alignment.
    if plan.bias == "neutral":
        q += 0.04
    elif (plan.bias == "up" and leg.direction == "long") or \
         (plan.bias == "down" and leg.direction == "short"):
        q += 0.10
    return float(min(1.0, max(0.0, q)))


def _signal_for_profile(
    cfg: Config,
    profile: FibProfile,
    df: pd.DataFrame,
    swings: list[Swing],
    plan: SessionPlan,
    session: Session,
    levels: list[Level],
    now: datetime,
    used_legs: dict[str, int],
    bar_index: int,
    atr_value: float,
    min_viable_stop: float,
) -> tuple[Signal | None, list[Rejection]]:
    rejections: list[Rejection] = []
    setup_name = f"fib_{profile.name}"
    last = len(df) - 1

    legs = find_impulse_legs(swings, profile, atr_value, last)
    if not legs:
        return None, rejections

    bar_high = float(df["high"].iloc[last])
    bar_low = float(df["low"].iloc[last])
    bar_close = float(df["close"].iloc[last])
    bar_open = float(df["open"].iloc[last])

    best: tuple[float, Signal] | None = None

    for leg in legs:
        key = f"{profile.name}:{leg.key()}"
        used = used_legs.get(key)
        if used is not None and bar_index - used < profile.leg_cooldown_bars:
            continue

        direction = leg.direction
        zone_a = leg.retracement(profile.entry_zone_start)
        zone_b = leg.retracement(profile.entry_zone_end)
        zone_hi, zone_lo = max(zone_a, zone_b), min(zone_a, zone_b)

        # The bar must have traded into the retracement zone.
        if bar_low > zone_hi or bar_high < zone_lo:
            continue

        invalidation = leg.retracement(profile.stop_level)
        if direction == "long" and bar_close < invalidation:
            continue     # retraced too deep, impulse failed
        if direction == "short" and bar_close > invalidation:
            continue

        # Confirmation: a close back in the impulse direction. Without it the
        # entry is a naked catch of a falling knife inside the zone.
        if profile.require_confirmation:
            confirmed = (bar_close > bar_open) if direction == "long" \
                else (bar_close < bar_open)
            if not confirmed:
                rejections.append(Rejection(now, session, setup_name,
                                            "awaiting_confirmation",
                                            {"leg": leg.key()}))
                continue

        if profile.require_bias_alignment and plan.bias != "neutral":
            aligned = (plan.bias == "up" and direction == "long") or \
                      (plan.bias == "down" and direction == "short")
            if not aligned:
                rejections.append(Rejection(now, session, setup_name,
                                            "bias_misaligned",
                                            {"bias": plan.bias, "dir": direction}))
                continue

        entry = bar_close

        # How deep the retracement actually went, as a fraction of the leg.
        depth = abs(leg.end_price - entry) / leg.length if leg.length > 0 else 0.0

        # --- stop ------------------------------------------------------------
        buf = profile.sl_buffer_atr * atr_value
        stop = invalidation - buf if direction == "long" else invalidation + buf
        stop_dist = abs(entry - stop)

        # Widen to whatever the current spread makes economically viable.
        floor = max(profile.min_stop_atr * atr_value, min_viable_stop)
        if stop_dist < floor:
            stop_dist = floor
            stop = entry - floor if direction == "long" else entry + floor

        if stop_dist > profile.max_stop_atr * atr_value:
            rejections.append(Rejection(now, session, setup_name, "stop_too_wide",
                                        {"stop_dist": round(stop_dist, 3),
                                         "atr": round(atr_value, 3),
                                         "leg": leg.key()}))
            continue

        # --- targets ----------------------------------------------------------
        tp1 = leg.extension(profile.tp1_extension)
        tp2 = leg.extension(profile.tp2_extension)

        # A target already behind price is not a target.
        if direction == "long" and tp1 <= entry:
            tp1 = entry + stop_dist
        if direction == "short" and tp1 >= entry:
            tp1 = entry - stop_dist
        if direction == "long" and tp2 <= tp1:
            tp2 = tp1 + stop_dist
        if direction == "short" and tp2 >= tp1:
            tp2 = tp1 - stop_dist

        # Do not aim through obvious opposing liquidity - continuation stalls there.
        target_level = opposing_liquidity(levels, entry, direction)
        if target_level is not None:
            pad = 0.15 * atr_value
            lvl = target_level.price - pad if direction == "long" \
                else target_level.price + pad
            capped = min(tp2, lvl) if direction == "long" else max(tp2, lvl)
            if abs(capped - entry) >= abs(tp1 - entry):
                tp2 = capped

        age = last - leg.end_index
        q = _quality(leg, profile, atr_value, plan, depth, age)

        sig = Signal(
            ts=now, session=session, setup=setup_name, direction=direction,
            entry=entry, stop=stop, tp1=tp1, tp2=tp2,
            reason=(f"[{profile.name}] {direction} leg "
                    f"{leg.start_price:.2f}->{leg.end_price:.2f} "
                    f"(${leg.length:.2f}) retraced to {depth:.3f}"),
            quality=q, level_name=f"fib_{profile.name}",
            meta={
                "profile": profile.name,
                "leg_key": key,
                "leg_start": leg.start_price,
                "leg_end": leg.end_price,
                "leg_length": round(leg.length, 3),
                "leg_bars": leg.bars,
                "retracement_depth": round(depth, 4),
                "entry_zone": [round(zone_lo, 2), round(zone_hi, 2)],
                "invalidation": round(invalidation, 2),
                "atr": atr_value,
                "risk_weight": profile.risk_weight,
                "target_level": target_level.name if target_level else None,
            },
        )
        if best is None or q > best[0]:
            best = (q, sig)

    return (best[1] if best else None), rejections


def find_fib_signals(
    cfg: Config,
    df: pd.DataFrame,
    plan: SessionPlan,
    session: Session,
    levels: list[Level],
    now: datetime,
    used_legs: dict[str, int],
    bar_index: int,
    min_viable_stop: float = 0.0,
) -> tuple[list[Signal], list[Rejection]]:
    """Evaluate every enabled profile. Returns one candidate per profile.

    Both profiles are returned rather than pre-selected between, so the engine's
    normal ranking and cost gate treat them on equal terms and the journal records
    what each would have done.
    """
    signals: list[Signal] = []
    rejections: list[Rejection] = []

    if not cfg.fib.enabled:
        return signals, rejections
    if "fib" not in plan.allowed_setups:
        return signals, rejections
    if len(df) < 80:
        return signals, rejections

    atr_value = float(df["atr"].iloc[-1])
    if atr_value <= 0 or pd.isna(atr_value):
        return signals, rejections

    swings = find_swings(df.tail(400), left=cfg.fib.swing_left,
                         right=cfg.fib.swing_right)
    if len(swings) < 2:
        return signals, rejections

    # find_swings indexes into the tail slice; shift back to full-frame indices.
    offset = len(df) - len(df.tail(400))
    swings = [Swing(s.index + offset, s.ts, s.price, s.kind) for s in swings]

    for profile in cfg.fib.profiles:
        if not profile.enabled:
            continue
        sig, rej = _signal_for_profile(
            cfg, profile, df, swings, plan, session, levels, now,
            used_legs, bar_index, atr_value, min_viable_stop,
        )
        rejections += rej
        if sig is not None:
            signals.append(sig)

    return signals, rejections