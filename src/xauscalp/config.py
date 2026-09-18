"""Configuration objects. Plain dataclasses loaded from YAML - no pydantic dependency."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


# --------------------------------------------------------------------------------------
# Sub-configs
# --------------------------------------------------------------------------------------
@dataclass
class SymbolCfg:
    name: str = "XAUUSD"
    # Fallbacks used only when MT5 symbol_info is unavailable (backtest / sim).
    fallback_point: float = 0.01          # smallest price increment
    fallback_contract_size: float = 100.0  # oz per 1.00 lot
    fallback_volume_min: float = 0.01
    fallback_volume_max: float = 50.0
    fallback_volume_step: float = 0.01
    fallback_digits: int = 2
    fallback_stops_level_points: int = 0   # broker min distance for SL/TP


@dataclass
class CostCfg:
    """Round-turn cost model. Gold: 1 lot = 100oz, so $1.00 price move = $100/lot.

    NOTE: some brokers (including MetaQuotes-Demo) report contract_size=100 but
    a tick_value implying 10oz. The bot uses tick_value, which is what MT5 uses
    for P/L. Commission must be scaled to match: ~0.35/lot/side on a 10oz
    contract, ~3.50 on a 100oz one. Run `python run.py --check` to see yours.
    """
    commission_per_lot_per_side: float = 3.5   # USD, raw/ECN typical
    assumed_slippage_points: float = 8.0       # points of adverse slippage per side
    max_spread_points: float = 45.0            # hard reject above this
    # Reject any setup where (round-turn cost) / (stop distance) exceeds this.
    # This also acts as a quality filter, not just an accounting one - lowering
    # it lets tighter, weaker setups through.
    max_cost_to_stop_ratio: float = 0.14
    # Reject any setup where (round-turn cost) / (TP1 distance) exceeds this.
    max_cost_to_target_ratio: float = 0.10


@dataclass
class SessionWindow:
    """Times are LOCAL to the given IANA tz, so DST is handled automatically."""
    tz: str = "UTC"
    start: str = "00:00"
    end: str = "07:00"
    enabled: bool = True


@dataclass
class SessionsCfg:
    asian: SessionWindow = field(
        default_factory=lambda: SessionWindow(tz="Asia/Tokyo", start="09:00", end="15:00")
    )
    london: SessionWindow = field(
        default_factory=lambda: SessionWindow(tz="Europe/London", start="08:00", end="16:30")
    )
    newyork: SessionWindow = field(
        default_factory=lambda: SessionWindow(tz="America/New_York", start="08:30", end="16:00")
    )
    # Range that defines Asian liquidity (local Tokyo time)
    asian_range_start: str = "09:00"
    asian_range_end: str = "14:00"
    # Minutes before session open at which the pre-session plan is generated
    plan_lead_minutes: int = 20
    # Do not open new trades in the final N minutes of a session
    no_new_trades_last_minutes: int = 45
    # Flatten everything N minutes before Friday close (broker/server time, UTC)
    friday_flatten_utc: str = "19:00"

    # DIRECTIONAL LOCK: decide the trend before the session opens, then take
    # ONLY that side for the whole session. No hedging, no flip-flopping.
    directional_lock: bool = True
    # Minimum vote score (0-1) required to commit to a direction. Below this the
    # session stands down rather than guessing. This is the single biggest
    # frequency lever in the system: 0.40 gives ~0.24 trades/day, 0.15 gives ~2.6.
    direction_min_conviction: float = 0.45
    # Minimum bars between entries, so a cluster of signals on one impulse does
    # not all pile into the same move.
    min_bars_between_entries: int = 3


@dataclass
class RegimeCfg:
    atr_period: int = 14
    atr_lookback_bars: int = 500        # window for ATR percentile ranking
    adx_period: int = 14
    # Score weights (sum need not be 1; normalised internally)
    w_volatility: float = 0.30
    w_trend: float = 0.25
    w_liquidity: float = 0.20   # spread health
    w_structure: float = 0.25   # clean swing structure / range integrity
    # Minimum regime score (0-100) to permit trading, per session
    min_score_asian: float = 52.0
    min_score_london: float = 50.0
    min_score_newyork: float = 75.0     # NY requires a genuinely strong read
    # Session-relative floor: quiet FOR THIS SESSION.
    atr_pct_floor: float = 0.15
    # Absolute floor across the whole day. Stops a session-relative score from
    # blessing a tape that cannot pay the spread in absolute terms.
    atr_pct_abs_floor: float = 0.06
    atr_pct_ceiling: float = 0.95       # too wild, stops get skipped


@dataclass
class SweepCfg:
    """Setup A: liquidity sweep + reclaim (mean reversion at session extremes)."""
    enabled: bool = True
    min_penetration_atr: float = 0.10   # wick must exceed level by this * ATR
    max_penetration_atr: float = 1.30   # beyond this it is a breakout, not a sweep
    reclaim_bars: int = 4               # bars allowed to close back inside
    sl_buffer_atr: float = 0.35
    tp1_r: float = 1.0
    tp2_r: float = 2.4
    min_stop_atr: float = 0.55          # never risk less than this * ATR (cost floor)
    max_stop_atr: float = 2.20
    require_opposing_liquidity: bool = True  # TP2 must have real liquidity to aim at
    level_cooldown_bars: int = 60       # do not re-trade the same level immediately


@dataclass
class DisplacementCfg:
    """Setup B: displacement + FVG retracement (momentum continuation)."""
    enabled: bool = True
    body_atr_mult: float = 1.35         # candle body must exceed this * ATR
    min_fvg_atr: float = 0.18           # gap must be economically meaningful
    max_fvg_age_bars: int = 12
    entry_fvg_fraction: float = 0.50    # 0 = near edge, 1 = far edge
    sl_buffer_atr: float = 0.30
    tp1_r: float = 1.0
    tp2_r: float = 2.6
    min_stop_atr: float = 0.60
    max_stop_atr: float = 2.50
    require_bias_alignment: bool = True


@dataclass
class OrbCfg:
    """Setup E - Opening Range Breakout. Fires roughly once per session."""
    enabled: bool = True
    orb_bars: int = 4              # M15 -> first hour defines the range
    min_range_atr: float = 0.7
    max_range_atr: float = 4.0
    break_buffer_atr: float = 0.12
    max_bars_after_open: int = 20  # 5h on M15 - the whole session
    min_stop_atr: float = 0.80
    max_stop_atr: float = 3.00
    tp1_range_mult: float = 0.8    # targets projected from the range size
    tp2_range_mult: float = 1.8


@dataclass
class VwapCfg:
    """Setup F - VWAP trend pullback. Repeats through a trending session."""
    enabled: bool = True
    touch_band_atr: float = 0.30
    max_extension_atr: float = 1.20
    min_bars_into_session: int = 6
    cooldown_bars: int = 6
    sl_buffer_atr: float = 0.35
    min_stop_atr: float = 0.80
    max_stop_atr: float = 3.00
    tp1_r: float = 1.0
    tp2_r: float = 2.2


@dataclass
class FibProfile:
    """One Fibonacci configuration. Two run in parallel so their results can be
    compared head to head in the journal rather than argued about."""
    name: str = "profile"
    enabled: bool = True

    # Retracement band that counts as a valid entry zone (fractions of the leg).
    entry_zone_start: float = 0.618
    entry_zone_end: float = 0.705

    # Retracement beyond which the impulse is considered failed.
    stop_level: float = 0.786
    sl_buffer_atr: float = 0.25

    # Targets as extensions of the leg. 1.0 = a return to the leg extreme.
    tp1_extension: float = 1.0
    tp2_extension: float = 1.618

    # Impulse quality
    min_leg_atr: float = 5.0        # leg length in ATR units
    max_leg_atr: float = 40.0
    max_leg_age_bars: int = 60
    min_leg_bars: int = 3
    max_leg_bars: int = 60

    require_confirmation: bool = True     # need a close back in the impulse direction
    require_bias_alignment: bool = True
    min_stop_atr: float = 0.50
    max_stop_atr: float = 3.00
    leg_cooldown_bars: int = 40
    risk_weight: float = 1.0              # scales position size for this profile


@dataclass
class FibCfg:
    enabled: bool = True
    swing_left: int = 3
    swing_right: int = 3
    profiles: list[FibProfile] = field(default_factory=lambda: [
        FibProfile(name="recommended"),
        FibProfile(name="user", entry_zone_start=0.50, entry_zone_end=0.618,
                   stop_level=1.0, tp1_extension=1.0, tp2_extension=1.272,
                   min_leg_atr=3.0, require_confirmation=False,
                   require_bias_alignment=False),
    ])


@dataclass
class RiskCfg:
    risk_pct_per_trade: float = 0.35        # % of equity risked at SL
    max_daily_loss_pct: float = 2.0
    max_weekly_loss_pct: float = 4.5
    max_daily_profit_pct: float = 4.0       # optional stop-at-green
    stop_on_daily_profit: bool = False
    max_concurrent_positions: int = 4
    max_trades_per_session: int = 10
    max_trades_per_day: int = 30

    # THE CAP THAT MAKES CONCURRENCY SAFE.
    # Concurrent same-direction gold positions are one leveraged bet, not
    # several independent ones - correlation is ~1.0. This limits the SUM of
    # open risk across every position. Per-trade size is reduced automatically
    # to stay inside it, so adding positions never increases total exposure.
    # A position whose stop has moved to break-even frees its budget back.
    max_portfolio_risk_pct: float = 1.2

    # Reject signals below this intrinsic quality even if trade slots remain.
    # The bot cannot manufacture good setups to hit a trade count; this stops it
    # filling the quota with rubbish.
    min_signal_quality: float = 0.40

    consecutive_loss_cooloff: int = 2       # after N losses, pause
    cooloff_minutes: int = 90
    # Risk is scaled by regime score: score 50 -> x0.6, score 100 -> x1.0
    scale_risk_by_regime: bool = True
    min_risk_scale: float = 0.5
    # Hard equity floor. Bot refuses to trade below this fraction of start equity.
    equity_floor_pct_of_start: float = 85.0


@dataclass
class ManageCfg:
    partial_at_tp1_pct: float = 50.0       # % of position closed at TP1
    move_to_be_at_r: float = 1.0
    be_offset_atr: float = 0.05
    trail_after_r: float = 1.3
    trail_atr_mult: float = 1.6            # chandelier distance
    time_stop_minutes: int = 90            # close if going nowhere
    time_stop_min_r: float = 0.3           # ...unless it has made at least this R
    max_hold_minutes: int = 240


@dataclass
class NewsCfg:
    enabled: bool = True
    csv_path: str = "config/news.csv"       # user-supplied calendar
    block_before_minutes: int = 20
    block_after_minutes: int = 25
    blocked_impacts: list[str] = field(default_factory=lambda: ["high"])
    # Widen guard: if spread suddenly exceeds N x its rolling median, treat as news
    spread_spike_multiple: float = 3.0


@dataclass
class ExecutionCfg:
    mode: str = "paper"                     # paper | live
    allow_live_on_real_account: bool = False  # extra seatbelt
    deviation_points: int = 30              # max slippage accepted on market orders
    magic: int = 770425
    comment: str = "xauscalp"
    poll_seconds: float = 2.0
    bar_timeframe: str = "M5"               # signal timeframe
    htf_timeframe: str = "M15"              # context timeframe
    bias_timeframe: str = "H1"              # bias timeframe
    history_bars: int = 1500
    retry_attempts: int = 3
    retry_sleep_seconds: float = 1.0
    # Broker server time minus UTC, in hours. null/None = auto-detect with
    # validation. Set it explicitly if auto-detection ever warns: a wrong
    # offset silently shifts every session boundary, and the bot keeps running
    # and keeps printing plans while believing London opens during Asia.
    server_utc_offset_hours: float | None = None


@dataclass
class BacktestCfg:
    start: str = "2025-01-01"
    end: str = "2026-08-01"
    initial_equity: float = 10000.0
    spread_points_mean: float = 22.0
    spread_points_std: float = 9.0
    slippage_points_mean: float = 6.0
    # Walk-forward
    train_days: int = 90
    test_days: int = 30


@dataclass
class Config:
    symbol: SymbolCfg = field(default_factory=SymbolCfg)
    costs: CostCfg = field(default_factory=CostCfg)
    sessions: SessionsCfg = field(default_factory=SessionsCfg)
    regime: RegimeCfg = field(default_factory=RegimeCfg)
    sweep: SweepCfg = field(default_factory=SweepCfg)
    displacement: DisplacementCfg = field(default_factory=DisplacementCfg)
    fib: FibCfg = field(default_factory=FibCfg)
    orb: OrbCfg = field(default_factory=OrbCfg)
    vwap: VwapCfg = field(default_factory=VwapCfg)
    risk: RiskCfg = field(default_factory=RiskCfg)
    manage: ManageCfg = field(default_factory=ManageCfg)
    news: NewsCfg = field(default_factory=NewsCfg)
    execution: ExecutionCfg = field(default_factory=ExecutionCfg)
    backtest: BacktestCfg = field(default_factory=BacktestCfg)
    db_path: str = "logs/journal.sqlite"
    log_path: str = "logs/xauscalp.log"
    log_level: str = "INFO"


# --------------------------------------------------------------------------------------
# Loader
# --------------------------------------------------------------------------------------
def _apply(obj: Any, data: dict) -> Any:
    """Recursively overlay a dict onto a dataclass instance."""
    if not is_dataclass(obj) or data is None:
        return obj
    known = {f.name: f for f in fields(obj)}
    for key, value in data.items():
        if key not in known:
            raise ValueError(f"Unknown config key: {key!r} in {type(obj).__name__}")
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _apply(current, value)
        elif isinstance(current, list) and isinstance(value, list) \
                and current and is_dataclass(current[0]):
            # A list of dataclasses (e.g. fib.profiles). Build each entry from
            # a fresh default so YAML only has to specify what it overrides.
            item_type = type(current[0])
            rebuilt = []
            for item in value:
                if isinstance(item, dict):
                    rebuilt.append(_apply(item_type(), item))
                else:
                    rebuilt.append(item)
            setattr(obj, key, rebuilt)
        else:
            setattr(obj, key, value)
    return obj


def load_config(path: str | Path | None = None) -> Config:
    cfg = Config()
    if path is None:
        path = os.environ.get("XAUSCALP_CONFIG", "config/default.yaml")
    p = Path(path)
    if p.exists():
        with p.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        _apply(cfg, raw)
    # Environment overrides for the dangerous switches only.
    if os.environ.get("XAUSCALP_MODE"):
        cfg.execution.mode = os.environ["XAUSCALP_MODE"].strip().lower()
    return cfg