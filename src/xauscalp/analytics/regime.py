"""Regime classification.

The single biggest determinant of whether an intraday gold strategy makes or
loses money is *when it is allowed to trade*, not the entry trigger. This module
produces a 0-100 score plus a categorical label; the strategy layer refuses to
act below a session-specific threshold, and the risk layer scales size by it.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd

from ..config import RegimeCfg
from ..sessions import Session
from .structure import find_swings, structure_bias


@dataclass
class RegimeRead:
    score: float
    label: str                # "trending" | "rotational" | "compressed" | "chaotic"
    atr: float
    atr_pct: float          # rank within this session's own history
    atr_pct_abs: float      # rank across the whole trading day
    adx: float
    spread_health: float      # 1.0 = tight, 0.0 = unusable
    structure: str            # "up" | "down" | "range"
    structure_quality: float
    tradable: bool
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def classify(
    df: pd.DataFrame,
    cfg: RegimeCfg,
    spread_points: float,
    median_spread_points: float,
    max_spread_points: float,
) -> RegimeRead:
    """df must already carry atr / atr_pct / adx columns (see indicators.enrich)."""
    reasons: list[str] = []

    if len(df) < max(cfg.atr_period * 3, 60):
        return RegimeRead(0.0, "compressed", 0.0, 0.0, 0.0, 0.0, 0.0, "range",
                          0.0, False, ["insufficient history"])

    atr_v = float(df["atr"].iloc[-1])
    # Session-relative rank: "is this busy for THIS session?"
    atr_pct = float(df["atr_pct"].iloc[-1]) if not np.isnan(df["atr_pct"].iloc[-1]) else 0.5
    # Absolute rank across the whole day, retained as a hard sanity floor so a
    # session-relative score cannot bless a tape that is dead in absolute terms.
    atr_pct_abs = float(df["atr_pct_global"].iloc[-1]) \
        if "atr_pct_global" in df.columns and not np.isnan(df["atr_pct_global"].iloc[-1]) \
        else atr_pct
    adx_v = float(df["adx"].iloc[-1])

    # ---- volatility component ---------------------------------------------------
    # Peak utility in the middle-upper band: enough movement to clear costs,
    # not so much that stops are gapped through.
    if atr_pct < cfg.atr_pct_floor or atr_pct_abs < cfg.atr_pct_abs_floor:
        vol_score = 0.0
        reasons.append(f"volatility too low (session={atr_pct:.2f}, "
                       f"absolute={atr_pct_abs:.2f})")
    elif atr_pct > cfg.atr_pct_ceiling:
        vol_score = 0.15
        reasons.append(f"volatility extreme (atr_pct={atr_pct:.2f})")
    else:
        span = cfg.atr_pct_ceiling - cfg.atr_pct_floor
        pos = (atr_pct - cfg.atr_pct_floor) / span if span > 0 else 0.5
        vol_score = _clip01(1.0 - abs(pos - 0.62) / 0.62)

    # ---- trend component --------------------------------------------------------
    trend_score = _clip01((adx_v - 12.0) / 26.0)

    # ---- liquidity component ----------------------------------------------------
    if spread_points <= 0:
        spread_health = 0.5
    elif spread_points >= max_spread_points:
        spread_health = 0.0
        reasons.append(f"spread {spread_points:.0f}pts >= max {max_spread_points:.0f}")
    else:
        ref = max(median_spread_points, 1.0)
        spread_health = _clip01(1.0 - (spread_points - ref) / max(max_spread_points - ref, 1.0))

    # ---- structure component ----------------------------------------------------
    swings = find_swings(df.tail(240), left=2, right=2)
    bias = structure_bias(df.tail(240), swings)
    # Quality = are swings well separated relative to ATR? Choppy tape gives
    # many swings crammed together, which is where reversal logic gets shredded.
    if len(swings) >= 4 and atr_v > 0:
        gaps = np.diff([s.index for s in swings])
        sep = float(np.median(gaps)) if len(gaps) else 0.0
        amp = float(np.median([abs(swings[i].price - swings[i - 1].price)
                               for i in range(1, len(swings))]))
        structure_quality = _clip01(min(sep / 6.0, 1.0) * min(amp / (1.2 * atr_v), 1.0))
    else:
        structure_quality = 0.3

    if structure_quality < 0.25:
        reasons.append("choppy structure")

    # ---- blend ------------------------------------------------------------------
    weights = np.array([cfg.w_volatility, cfg.w_trend, cfg.w_liquidity, cfg.w_structure])
    weights = weights / weights.sum()
    parts = np.array([vol_score, trend_score, spread_health, structure_quality])
    score = float((weights * parts).sum() * 100.0)

    # ---- label ------------------------------------------------------------------
    if atr_pct > cfg.atr_pct_ceiling:
        label = "chaotic"
    elif atr_pct < cfg.atr_pct_floor:
        label = "compressed"
    elif adx_v >= 24.0 and bias in ("up", "down"):
        label = "trending"
    else:
        label = "rotational"

    tradable = (spread_health > 0.0
                and atr_pct >= cfg.atr_pct_floor
                and atr_pct_abs >= cfg.atr_pct_abs_floor)
    if not tradable and not reasons:
        reasons.append("regime not tradable")

    return RegimeRead(
        score=round(score, 1),
        label=label,
        atr=round(atr_v, 4),
        atr_pct=round(atr_pct, 3),
        atr_pct_abs=round(atr_pct_abs, 3),
        adx=round(adx_v, 1),
        spread_health=round(spread_health, 3),
        structure=bias,
        structure_quality=round(structure_quality, 3),
        tradable=tradable,
        reasons=reasons,
    )


def min_score_for(cfg: RegimeCfg, session: Session) -> float:
    return {
        Session.ASIAN: cfg.min_score_asian,
        Session.LONDON: cfg.min_score_london,
        Session.NEWYORK: cfg.min_score_newyork,
    }.get(session, 101.0)


def preferred_setups(read: RegimeRead, session: Session) -> list[str]:
    """Which setup families make sense given the regime.

    Rotational tape -> fade the extremes. Trending tape -> join the displacement.
    Getting this backwards is the most common way these systems bleed.
    """
    if read.label == "chaotic":
        return []
    if read.label == "compressed":
        # Nothing has moved far enough to project a Fibonacci leg from, but a
        # compressed session is exactly when an opening range is worth watching.
        return ["sweep", "orb"] if session == Session.ASIAN else ["orb"]
    if read.label == "trending":
        # Trending tape is where retracement continuation belongs.
        if session == Session.ASIAN:
            return ["sweep", "fib", "orb", "vwap"]
        return ["displacement", "fib", "vwap", "orb", "sweep"]
    # rotational: retracements are less reliable, but the golden-pocket entry
    # still works at range extremes, so fib stays enabled at lower priority.
    if session == Session.ASIAN:
        return ["sweep", "fib", "orb"]
    return ["sweep", "displacement", "fib", "orb", "vwap"]