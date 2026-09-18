"""Market structure primitives.

These encode the mechanics that stop-hunting / liquidity models describe: swing
points are where resting stops cluster, a sweep is a raid on those stops, a fair
value gap is an imbalance left by aggressive one-sided flow, and displacement is
the footprint of size entering the book.

Nothing here is predictive on its own. These are *feature extractors*; the edge
hypothesis lives in strategy/, and the cost filter in risk/ decides whether any
of it is economically tradeable.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------
# Swings
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Swing:
    index: int
    ts: datetime
    price: float
    kind: str  # "high" | "low"


def find_swings(df: pd.DataFrame, left: int = 2, right: int = 2) -> list[Swing]:
    """Fractal swing points. `right` bars must have elapsed, so a swing is only
    confirmed `right` bars after it printed - no lookahead."""
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    n = len(df)
    out: list[Swing] = []
    for i in range(left, n - right):
        window_h = highs[i - left: i + right + 1]
        window_l = lows[i - left: i + right + 1]
        if highs[i] == window_h.max() and (window_h.argmax() == left):
            out.append(Swing(i, df.index[i].to_pydatetime(), float(highs[i]), "high"))
        if lows[i] == window_l.min() and (window_l.argmin() == left):
            out.append(Swing(i, df.index[i].to_pydatetime(), float(lows[i]), "low"))
    out.sort(key=lambda s: s.index)
    return out


def last_swing(swings: list[Swing], kind: str, before_index: int | None = None) -> Swing | None:
    for s in reversed(swings):
        if s.kind != kind:
            continue
        if before_index is not None and s.index >= before_index:
            continue
        return s
    return None


def structure_bias(df: pd.DataFrame, swings: list[Swing], lookback: int = 6) -> str:
    """Higher-highs/higher-lows -> up, lower-lows/lower-highs -> down, else range."""
    highs = [s for s in swings if s.kind == "high"][-lookback:]
    lows = [s for s in swings if s.kind == "low"][-lookback:]
    if len(highs) < 2 or len(lows) < 2:
        return "range"
    hh = highs[-1].price > highs[-2].price
    hl = lows[-1].price > lows[-2].price
    lh = highs[-1].price < highs[-2].price
    ll = lows[-1].price < lows[-2].price
    if hh and hl:
        return "up"
    if lh and ll:
        return "down"
    return "range"


def choch(df: pd.DataFrame, swings: list[Swing], direction: str,
          from_index: int) -> bool:
    """Change of character: after `from_index`, price closes beyond the most recent
    opposing swing, i.e. the prior structure has been broken."""
    if direction == "long":
        ref = last_swing(swings, "high", before_index=from_index)
        if ref is None:
            return False
        return bool((df["close"].iloc[from_index:] > ref.price).any())
    ref = last_swing(swings, "low", before_index=from_index)
    if ref is None:
        return False
    return bool((df["close"].iloc[from_index:] < ref.price).any())


# --------------------------------------------------------------------------------------
# Fair value gaps (3-bar imbalance)
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class FVG:
    index: int          # index of the third (confirming) bar
    ts: datetime
    top: float
    bottom: float
    direction: str      # "bull" | "bear"

    @property
    def size(self) -> float:
        return self.top - self.bottom

    def midpoint(self, fraction: float = 0.5) -> float:
        """fraction=0 -> the edge price first touched on a retrace into the gap."""
        if self.direction == "bull":
            return self.top - fraction * self.size
        return self.bottom + fraction * self.size


def find_fvgs(df: pd.DataFrame, min_size: float = 0.0, lookback: int = 200) -> list[FVG]:
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    n = len(df)
    start = max(2, n - lookback)
    out: list[FVG] = []
    for i in range(start, n):
        # Bullish imbalance: low of bar i sits above high of bar i-2
        if lows[i] > highs[i - 2]:
            size = lows[i] - highs[i - 2]
            if size >= min_size:
                out.append(FVG(i, df.index[i].to_pydatetime(),
                               float(lows[i]), float(highs[i - 2]), "bull"))
        # Bearish imbalance: high of bar i sits below low of bar i-2
        if highs[i] < lows[i - 2]:
            size = lows[i - 2] - highs[i]
            if size >= min_size:
                out.append(FVG(i, df.index[i].to_pydatetime(),
                               float(lows[i - 2]), float(highs[i]), "bear"))
    return out


def fvg_is_fresh(fvg: FVG, df: pd.DataFrame) -> bool:
    """A gap that has already been fully traded through carries no information."""
    after = df.iloc[fvg.index + 1:]
    if after.empty:
        return True
    if fvg.direction == "bull":
        return bool(after["low"].min() > fvg.bottom)
    return bool(after["high"].max() < fvg.top)


# --------------------------------------------------------------------------------------
# Displacement
# --------------------------------------------------------------------------------------
def is_displacement(df: pd.DataFrame, i: int, atr_value: float, body_mult: float) -> str | None:
    """Returns 'bull'/'bear' if bar i is an outsized directional candle."""
    if atr_value <= 0 or i < 0 or i >= len(df):
        return None
    o = float(df["open"].iloc[i])
    c = float(df["close"].iloc[i])
    rng = float(df["high"].iloc[i] - df["low"].iloc[i])
    b = abs(c - o)
    if b < body_mult * atr_value:
        return None
    if rng > 0 and b / rng < 0.55:   # mostly wick = indecision, not displacement
        return None
    return "bull" if c > o else "bear"


# --------------------------------------------------------------------------------------
# Sweeps
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Sweep:
    index: int
    ts: datetime
    level: float
    level_name: str
    extreme: float      # the wick extreme of the sweep bar
    direction: str      # "long" = swept a low, expecting reversal up
    penetration: float


def detect_sweep(
    df: pd.DataFrame,
    i: int,
    level: float,
    level_name: str,
    atr_value: float,
    min_pen_atr: float,
    max_pen_atr: float,
    level_kind: str = "both",
) -> Sweep | None:
    """Bar i raids `level` with its wick but closes back on the origin side.

    That close-back is the whole point: a bar that penetrates and *holds* is a
    breakout, and trading it as a reversal is how sweep models lose money.

    `level_kind` must be "high" (resistance - stops rest above it, so a raid
    implies a short) or "low" (support - implies a long). Checking both sides of
    a single level is wrong: a wide bar straddling the level would register as a
    sweep in whichever direction happened to match, which is noise, not signal.
    """
    if atr_value <= 0:
        return None
    high = float(df["high"].iloc[i])
    low = float(df["low"].iloc[i])
    close = float(df["close"].iloc[i])
    min_pen = min_pen_atr * atr_value
    max_pen = max_pen_atr * atr_value

    # Swept a high -> look for a short
    if level_kind in ("high", "both"):
        if high > level + min_pen and close < level:
            pen = high - level
            if pen <= max_pen:
                return Sweep(i, df.index[i].to_pydatetime(), level, level_name,
                             high, "short", pen)
    # Swept a low -> look for a long
    if level_kind in ("low", "both"):
        if low < level - min_pen and close > level:
            pen = level - low
            if pen <= max_pen:
                return Sweep(i, df.index[i].to_pydatetime(), level, level_name,
                             low, "long", pen)
    return None


def reclaim_confirmed(df: pd.DataFrame, sweep: Sweep, bars: int) -> int | None:
    """Index of the bar that confirms the reversal, or None.

    Confirmation = a close beyond the swept level in the reversal direction with
    the sweep extreme still intact.
    """
    end = min(len(df), sweep.index + bars + 1)
    for j in range(sweep.index + 1, end):
        c = float(df["close"].iloc[j])
        if sweep.direction == "long":
            if float(df["low"].iloc[j]) < sweep.extreme:
                return None  # extreme broken, thesis invalid
            if c > sweep.level:
                return j
        else:
            if float(df["high"].iloc[j]) > sweep.extreme:
                return None
            if c < sweep.level:
                return j
    return None