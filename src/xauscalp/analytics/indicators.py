"""Indicator primitives. Pure numpy/pandas, no TA-Lib compile step required."""
from __future__ import annotations

import numpy as np
import pandas as pd


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    a = df["high"] - df["low"]
    b = (df["high"] - prev_close).abs()
    c = (df["low"] - prev_close).abs()
    return pd.concat([a, b, c], axis=1).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's ATR (RMA smoothing), not the SMA approximation."""
    tr = true_range(df)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.fillna(50.0)


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder ADX. Used only as a regime descriptor, never as a standalone signal."""
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = true_range(df)
    atr_ = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    plus_di = 100.0 * pd.Series(plus_dm, index=df.index).ewm(
        alpha=1.0 / period, adjust=False, min_periods=period
    ).mean() / atr_.replace(0.0, np.nan)
    minus_di = 100.0 * pd.Series(minus_dm, index=df.index).ewm(
        alpha=1.0 / period, adjust=False, min_periods=period
    ).mean() / atr_.replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean().fillna(0.0)


def rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    """Fraction of the trailing window that the current value exceeds (0..1)."""
    def _pct(x: np.ndarray) -> float:
        if len(x) < 2:
            return 0.5
        return float((x[:-1] < x[-1]).mean())

    return series.rolling(window, min_periods=max(20, window // 10)).apply(_pct, raw=True)


def session_vwap(df: pd.DataFrame, session_start_mask: pd.Series) -> pd.Series:
    """VWAP anchored to each session start. Uses tick_volume as the weight."""
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df.get("tick_volume")
    if vol is None:
        vol = pd.Series(1.0, index=df.index)
    vol = vol.replace(0, 1.0).astype(float)
    group = session_start_mask.cumsum()
    pv = (tp * vol).groupby(group).cumsum()
    vv = vol.groupby(group).cumsum()
    return pv / vv.replace(0.0, np.nan)


def realized_vol(series: pd.Series, period: int = 30) -> pd.Series:
    """Annualisation is meaningless intraday; this is raw bar-return stdev."""
    return series.pct_change().rolling(period, min_periods=period // 2).std()


def body(df: pd.DataFrame) -> pd.Series:
    return (df["close"] - df["open"]).abs()


def upper_wick(df: pd.DataFrame) -> pd.Series:
    return df["high"] - df[["open", "close"]].max(axis=1)


def lower_wick(df: pd.DataFrame) -> pd.Series:
    return df[["open", "close"]].min(axis=1) - df["low"]


def session_bucket(index: pd.DatetimeIndex) -> pd.Series:
    """Coarse session label per bar, from the UTC hour.

    Deliberately approximate (it ignores DST by an hour) because it is only used
    to normalise volatility statistics, not to make trading decisions. The real
    session clock in sessions.py is DST-exact.
    """
    hours = index.hour
    out = np.select(
        [
            (hours >= 0) & (hours < 7),
            (hours >= 7) & (hours < 12),
            (hours >= 12) & (hours < 20),
        ],
        ["asian", "london", "newyork"],
        default="off",
    )
    return pd.Series(out, index=index)


def session_relative_percentile(series: pd.Series, buckets: pd.Series,
                                window: int) -> pd.Series:
    """Rank each value against the trailing history of the SAME session only.

    This matters more than it looks. Ranked globally, every Asian bar sits in the
    bottom decile of the ATR distribution simply because London and New York bars
    dominate it - so a volatility filter calibrated on the whole day declares the
    Asian session permanently dead and the strategy never trades it. Ranking
    within session asks the question that actually matters: is this Asian session
    busy or quiet *for an Asian session*?
    """
    out = pd.Series(np.nan, index=series.index, dtype=float)
    per_session_window = max(30, window // 3)
    for name in buckets.unique():
        mask = (buckets == name).to_numpy()
        if mask.sum() < 20:
            continue
        sub = series[mask]
        out.loc[sub.index] = rolling_percentile(sub, per_session_window)
    return out.ffill().fillna(0.5)


def enrich(df: pd.DataFrame, atr_period: int = 14, adx_period: int = 14,
           atr_lookback: int = 500) -> pd.DataFrame:
    """Attach the standard indicator set. Returns a copy."""
    out = df.copy()
    out["atr"] = atr(out, atr_period)
    out["atr_pct_global"] = rolling_percentile(out["atr"], atr_lookback)
    out["session_bucket"] = session_bucket(out.index)
    out["atr_pct"] = session_relative_percentile(
        out["atr"], out["session_bucket"], atr_lookback
    )
    out["adx"] = adx(out, adx_period)
    out["rsi"] = rsi(out["close"], 14)
    out["ema_fast"] = ema(out["close"], 21)
    out["ema_slow"] = ema(out["close"], 55)
    out["body"] = body(out)
    out["upper_wick"] = upper_wick(out)
    out["lower_wick"] = lower_wick(out)
    return out