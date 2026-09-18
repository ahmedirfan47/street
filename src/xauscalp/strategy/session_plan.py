"""Pre-session analysis.

Runs `plan_lead_minutes` before each session open and produces a SessionPlan:
the regime read, directional bias, which setup families are permitted, the level
map, and a risk multiplier. The live loop will not take a trade unless the
current session has a plan that permits that setup family.

This is deliberately the same code path in backtest and live, so what you test
is what you run.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from ..analytics import regime as regime_mod
from ..analytics.indicators import enrich
from ..analytics.levels import build_levels
from ..analytics.structure import find_swings, structure_bias
from ..config import Config
from ..sessions import Session, SessionClock
from .base import SessionPlan


def _bias_from_htf(htf: pd.DataFrame) -> str:
    """Directional lean from the higher timeframe. Deliberately conservative:
    it returns 'neutral' unless structure and the EMA stack agree."""
    if len(htf) < 60:
        return "neutral"
    swings = find_swings(htf.tail(200), left=2, right=2)
    struct = structure_bias(htf.tail(200), swings)
    fast = htf["ema_fast"].iloc[-1]
    slow = htf["ema_slow"].iloc[-1]
    if pd.isna(fast) or pd.isna(slow):
        return "neutral"
    ema_dir = "up" if fast > slow else "down"
    if struct == ema_dir:
        return struct
    return "neutral"


def determine_session_direction(
    cfg: Config, sig: pd.DataFrame, htf: pd.DataFrame, adx_value: float,
) -> tuple[str, float, str]:
    """Recognise the trend BEFORE deciding to buy or sell.

    Five weighted votes, each in -1..+1, blended into a single score. The sign
    picks the side; the magnitude becomes conviction. ADX then scales conviction
    down when the tape is directionless - a strong structural read in a chop
    market is still a chop market.

    Returns (direction, conviction, reason). "none" means stand down rather
    than guess, which is the correct answer more often than people like.
    """
    votes: list[tuple[str, float, float]] = []
    atr_h = float(htf["atr"].iloc[-1]) if len(htf) else 0.0
    if atr_h <= 0:
        return "none", 0.0, "no ATR"

    # 1. Higher-timeframe market structure - the strongest single signal.
    swings = find_swings(htf.tail(200), left=2, right=2)
    struct = structure_bias(htf.tail(200), swings)
    votes.append(("struct", 0.30,
                  1.0 if struct == "up" else -1.0 if struct == "down" else 0.0))

    # 2. HTF EMA stack, scaled by separation in ATR units. A 0.1-ATR gap is
    #    noise; a 2-ATR gap is a committed trend.
    fast = float(htf["ema_fast"].iloc[-1])
    slow = float(htf["ema_slow"].iloc[-1])
    if not (pd.isna(fast) or pd.isna(slow)):
        sep = (fast - slow) / atr_h
        votes.append(("ema_htf", 0.25, max(-1.0, min(1.0, sep / 1.5))))

    # 3. Slope of the signal-timeframe fast EMA over the last 10 bars.
    if len(sig) > 12 and not pd.isna(sig["ema_fast"].iloc[-1]):
        atr_s = float(sig["atr"].iloc[-1])
        if atr_s > 0:
            slope = (float(sig["ema_fast"].iloc[-1])
                     - float(sig["ema_fast"].iloc[-11])) / atr_s
            votes.append(("slope", 0.20, max(-1.0, min(1.0, slope / 2.0))))

    # 4. Where price sits in its recent range. Near the highs favours longs.
    look = sig.tail(120)
    if len(look) > 20:
        hi, lo = float(look["high"].max()), float(look["low"].min())
        if hi > lo:
            pos = (float(sig["close"].iloc[-1]) - lo) / (hi - lo)
            votes.append(("range_pos", 0.15, (pos - 0.5) * 2.0))

    # 5. HTF close versus its own 50-bar mean.
    if len(htf) > 55:
        mean50 = float(htf["close"].tail(50).mean())
        dev = (float(htf["close"].iloc[-1]) - mean50) / atr_h
        votes.append(("mean_dev", 0.10, max(-1.0, min(1.0, dev / 3.0))))

    total_w = sum(w for _, w, _ in votes)
    if total_w <= 0:
        return "none", 0.0, "no votes"
    score = sum(w * v for _, w, v in votes) / total_w

    # ADX scales conviction: below ~15 the market has no trend to follow.
    adx_factor = max(0.25, min(1.0, (adx_value - 12.0) / 20.0))
    conviction = abs(score) * adx_factor

    detail = " ".join(f"{n}{v:+.2f}" for n, _, v in votes)
    reason = f"score{score:+.2f} adx{adx_value:.0f}x{adx_factor:.2f} | {detail}"

    if conviction < cfg.sessions.direction_min_conviction:
        return "none", conviction, reason + " -> below conviction floor"
    return ("long" if score > 0 else "short"), conviction, reason


def build_session_plan(
    cfg: Config,
    now: datetime,
    session: Session,
    session_open_utc: datetime,
    m5: pd.DataFrame,
    htf: pd.DataFrame,
    spread_points: float,
    median_spread_points: float,
) -> SessionPlan:
    notes: list[str] = []

    m5e = enrich(m5, cfg.regime.atr_period, cfg.regime.adx_period, cfg.regime.atr_lookback_bars)
    htfe = enrich(htf, cfg.regime.atr_period, cfg.regime.adx_period, cfg.regime.atr_lookback_bars)

    read = regime_mod.classify(
        m5e, cfg.regime, spread_points, median_spread_points, cfg.costs.max_spread_points
    )
    bias = _bias_from_htf(htfe)

    clock = SessionClock(cfg.sessions)
    swings = find_swings(m5e.tail(400), left=2, right=2)
    level_objs = build_levels(
        m5e, clock, now,
        swing_highs=[s.price for s in swings if s.kind == "high"],
        swing_lows=[s.price for s in swings if s.kind == "low"],
    )

    # Trend first, then direction. Nothing is permitted until this resolves.
    if cfg.sessions.directional_lock:
        direction, conviction, dir_reason = determine_session_direction(
            cfg, m5e, htfe, read.adx
        )
    else:
        direction, conviction, dir_reason = "both", 1.0, "directional lock off"

    allowed = regime_mod.preferred_setups(read, session)
    if not cfg.sweep.enabled and "sweep" in allowed:
        allowed.remove("sweep")
    if not cfg.displacement.enabled and "displacement" in allowed:
        allowed.remove("displacement")
    if not cfg.orb.enabled and "orb" in allowed:
        allowed.remove("orb")
    if not cfg.vwap.enabled and "vwap" in allowed:
        allowed.remove("vwap")
    if "fib" in allowed:
        active_profiles = [p.name for p in cfg.fib.profiles if p.enabled]
        if not cfg.fib.enabled or not active_profiles:
            allowed.remove("fib")
        else:
            notes.append("Fib profiles active: " + ", ".join(active_profiles))

    min_score = regime_mod.min_score_for(cfg.regime, session)
    tradable = (read.tradable and read.score >= min_score and bool(allowed)
                and direction != "none")

    if direction == "none":
        notes.append(f"No directional conviction ({conviction:.2f} < "
                     f"{cfg.sessions.direction_min_conviction:.2f}) - stand down")
    elif direction in ("long", "short"):
        notes.append(f"SESSION LOCKED {direction.upper()}-ONLY "
                     f"(conviction {conviction:.2f}). Opposite-side setups "
                     f"will be rejected all session.")

    if not read.tradable:
        notes.append("Regime flagged untradable: " + "; ".join(read.reasons))
    if read.score < min_score:
        notes.append(f"Score {read.score:.1f} below {session.value} threshold {min_score:.0f}")
    if session == Session.NEWYORK and tradable:
        notes.append("NY session cleared - required an elevated regime score to qualify")
    if not level_objs:
        notes.append("No key levels resolved; sweep engine will have nothing to work with")
        tradable = False

    # Risk scaling: linear from the session threshold up to 100.
    if cfg.risk.scale_risk_by_regime and tradable:
        span = max(100.0 - min_score, 1.0)
        frac = max(0.0, min(1.0, (read.score - min_score) / span))
        risk_scale = cfg.risk.min_risk_scale + frac * (1.0 - cfg.risk.min_risk_scale)
    else:
        risk_scale = 1.0 if tradable else 0.0

    max_trades = cfg.risk.max_trades_per_session if tradable else 0

    if read.label == "trending" and bias != "neutral":
        notes.append(f"Trending tape with {bias} bias - counter-trend sweeps require "
                     f"a stronger reclaim")
    if read.label == "rotational":
        notes.append("Rotational tape - range extremes favoured, continuation deprioritised")

    return SessionPlan(
        created_at=now,
        session=session,
        session_open_utc=session_open_utc,
        regime=read,
        bias=bias,
        allowed_setups=allowed if tradable else [],
        key_levels=[
            {"name": l.name, "price": round(l.price, 2), "kind": l.kind, "weight": l.weight}
            for l in sorted(level_objs, key=lambda x: -x.weight)
        ],
        risk_scale=round(risk_scale, 3),
        max_trades=max_trades,
        direction=direction,
        direction_conviction=round(conviction, 3),
        direction_reason=dir_reason,
        notes=notes,
        tradable=tradable,
    )