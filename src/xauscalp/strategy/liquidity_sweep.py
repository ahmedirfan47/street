"""Setup A - Liquidity Sweep Reversal.

Hypothesis: resting stop orders cluster immediately beyond session and prior-day
extremes. Price frequently trades through those extremes, fills the resting
liquidity, and then fails to hold - because the move was liquidity-seeking
rather than informed. The tradable event is the *failure*, evidenced by a close
back inside the level with the sweep extreme intact.

Risk profile: this is a counter-trend entry. It is gated hard - it will not fire
into a strong trending tape in the opposite direction without a confirmed
change of character, and it requires a real opposing level to target.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from ..analytics.levels import Level, opposing_liquidity
from ..analytics.structure import Sweep, detect_sweep, find_swings, reclaim_confirmed
from ..config import Config
from ..sessions import Session
from .base import Rejection, Signal, SessionPlan


def _quality(sweep: Sweep, level: Level, atr_v: float, plan: SessionPlan,
             reclaim_bars_used: int, cfg: Config) -> float:
    """0..1. Rewards heavyweight levels, decisive reclaims, clean penetration."""
    q = 0.30 * level.weight

    # Penetration in the sweet spot: deep enough to have taken stops, shallow
    # enough that it was not a genuine breakout.
    pen_atr = sweep.penetration / atr_v if atr_v > 0 else 0.0
    ideal = 0.45
    q += 0.25 * max(0.0, 1.0 - abs(pen_atr - ideal) / ideal)

    # Faster reclaim = more decisive rejection.
    q += 0.20 * max(0.0, 1.0 - (reclaim_bars_used - 1) / max(cfg.sweep.reclaim_bars, 1))

    # Regime score contribution.
    q += 0.15 * min(1.0, plan.regime.score / 100.0)

    # Bias alignment: reward trading with the higher-timeframe lean.
    if plan.bias == "neutral":
        q += 0.05
    elif (plan.bias == "up" and sweep.direction == "long") or \
         (plan.bias == "down" and sweep.direction == "short"):
        q += 0.10
    return float(min(1.0, max(0.0, q)))


def find_sweep_signal(
    cfg: Config,
    df: pd.DataFrame,
    plan: SessionPlan,
    session: Session,
    levels: list[Level],
    now: datetime,
    recent_level_use: dict[str, int],
    bar_index: int,
    min_viable_stop: float = 0.0,
) -> tuple[Signal | None, list[Rejection]]:
    """Evaluate the most recent bars for a completed sweep + reclaim.

    Returns (signal, rejections). Only ever returns a signal on the bar that
    confirms the reclaim, so there is no lookahead.

    `min_viable_stop` is the smallest stop distance that can clear current
    round-turn costs at the configured cost ratio. Stops are widened to it
    rather than the setup being silently rejected downstream - a structurally
    valid sweep with a stop that is merely too tight for today's spread is a
    sizing problem, not an invalid setup.
    """
    rejections: list[Rejection] = []
    scfg = cfg.sweep

    if "sweep" not in plan.allowed_setups:
        return None, rejections
    if len(df) < 60:
        return None, rejections

    atr_v = float(df["atr"].iloc[-1])
    if atr_v <= 0 or pd.isna(atr_v):
        return None, rejections

    confirm_idx = len(df) - 1  # we only act on the just-closed bar

    best: tuple[float, Signal] | None = None

    # Look back over the window in which a sweep could still be reclaimed now.
    earliest = max(1, confirm_idx - scfg.reclaim_bars)
    for i in range(earliest, confirm_idx):
        for level in levels:
            key = f"{level.name}:{level.price:.2f}"
            last_used = recent_level_use.get(key)
            if last_used is not None and bar_index - last_used < scfg.level_cooldown_bars:
                continue

            sweep = detect_sweep(
                df, i, level.price, level.name, atr_v,
                scfg.min_penetration_atr, scfg.max_penetration_atr,
                level_kind=level.kind,
            )
            if sweep is None:
                continue

            j = reclaim_confirmed(df, sweep, scfg.reclaim_bars)
            if j is None or j != confirm_idx:
                continue  # not confirmed, or confirmed on an earlier bar already handled

            direction = sweep.direction
            entry = float(df["close"].iloc[confirm_idx])

            # --- stop: beyond the sweep extreme -------------------------------
            buf = scfg.sl_buffer_atr * atr_v
            stop = sweep.extreme - buf if direction == "long" else sweep.extreme + buf
            stop_dist = abs(entry - stop)

            # Floor the stop at whichever is larger: the structural minimum, or
            # the distance required for the trade to clear its own costs.
            floor = max(scfg.min_stop_atr * atr_v, min_viable_stop)
            if stop_dist < floor:
                stop_dist = floor
                stop = entry - floor if direction == "long" else entry + floor

            if stop_dist > scfg.max_stop_atr * atr_v:
                rejections.append(Rejection(now, session, "sweep", "stop_too_wide",
                                            {"stop_dist": stop_dist, "atr": atr_v,
                                             "level": level.name}))
                continue

            # --- target: must be real liquidity, not an invented R multiple ----
            target_level = opposing_liquidity(levels, entry, direction)
            if scfg.require_opposing_liquidity and target_level is None:
                rejections.append(Rejection(now, session, "sweep", "no_opposing_liquidity",
                                            {"level": level.name}))
                continue

            tp1 = entry + (scfg.tp1_r * stop_dist if direction == "long"
                           else -scfg.tp1_r * stop_dist)
            tp2_r_price = entry + (scfg.tp2_r * stop_dist if direction == "long"
                                   else -scfg.tp2_r * stop_dist)
            if target_level is not None:
                # Aim just short of the level; the last few points are where
                # fills get ugly.
                pad = 0.15 * atr_v
                lvl_target = (target_level.price - pad if direction == "long"
                              else target_level.price + pad)
                tp2 = (min(tp2_r_price, lvl_target) if direction == "long"
                       else max(tp2_r_price, lvl_target))
            else:
                tp2 = tp2_r_price

            if abs(tp2 - entry) < abs(tp1 - entry):
                rejections.append(Rejection(now, session, "sweep", "target_inside_tp1",
                                            {"level": level.name}))
                continue

            # --- counter-trend guard ------------------------------------------
            if plan.regime.label == "trending":
                against = ((plan.bias == "up" and direction == "short") or
                           (plan.bias == "down" and direction == "long"))
                if against:
                    swings = find_swings(df.tail(120), left=2, right=2)
                    from ..analytics.structure import choch
                    if not choch(df, swings, direction, sweep.index):
                        rejections.append(Rejection(
                            now, session, "sweep", "counter_trend_without_choch",
                            {"bias": plan.bias, "direction": direction}))
                        continue

            reclaim_bars_used = j - sweep.index
            q = _quality(sweep, level, atr_v, plan, reclaim_bars_used, cfg)

            sig = Signal(
                ts=now, session=session, setup="sweep", direction=direction,
                entry=entry, stop=stop, tp1=tp1, tp2=tp2,
                reason=(f"swept {level.name} @{level.price:.2f} by "
                        f"{sweep.penetration:.2f} then reclaimed in "
                        f"{reclaim_bars_used} bar(s)"),
                quality=q, level_name=level.name,
                meta={"level_price": level.price, "sweep_extreme": sweep.extreme,
                      "penetration_atr": round(sweep.penetration / atr_v, 3),
                      "atr": atr_v, "level_key": key,
                      "target_level": target_level.name if target_level else None},
            )
            if best is None or q > best[0]:
                best = (q, sig)

    return (best[1] if best else None), rejections