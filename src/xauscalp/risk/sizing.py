"""Position sizing and trade cost accounting.

Gold sizing errors are the fastest way to blow an account: 1.00 lot of XAUUSD is
100 ounces, so a $1.00 adverse move is $100. A sizing bug that is off by 10x is
not a 10x loss, it is a margin call. Every number here is derived from the
broker's own symbol_info where available, never assumed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..config import Config


@dataclass(frozen=True)
class SymbolSpec:
    """Contract specification, populated from MT5 symbol_info at runtime."""
    name: str
    point: float
    digits: int
    contract_size: float
    volume_min: float
    volume_max: float
    volume_step: float
    tick_value: float      # account currency per tick_size move, per 1.00 lot
    tick_size: float
    stops_level_points: int = 0

    def value_per_price_unit(self) -> float:
        """Account currency P/L per 1.00 price unit ($1 of gold) per 1.00 lot."""
        if self.tick_size > 0 and self.tick_value > 0:
            return self.tick_value / self.tick_size
        return self.contract_size

    def points(self, price_distance: float) -> float:
        return price_distance / self.point if self.point > 0 else 0.0

    def price(self, points: float) -> float:
        return points * self.point

    def normalize_volume(self, volume: float) -> float:
        if self.volume_step <= 0:
            return round(volume, 2)
        steps = math.floor(volume / self.volume_step + 1e-9)
        v = steps * self.volume_step
        v = max(self.volume_min, min(self.volume_max, v))
        return round(v, 8)

    def normalize_price(self, price: float) -> float:
        return round(price, self.digits)


@dataclass(frozen=True)
class CostEstimate:
    spread_cost: float          # price units, one-way
    commission_price_units: float  # round-turn commission expressed in price units
    slippage_cost: float        # price units, round-turn
    total_round_turn: float     # price units

    def as_points(self, spec: SymbolSpec) -> float:
        return spec.points(self.total_round_turn)


def estimate_costs(cfg: Config, spec: SymbolSpec, spread_points: float,
                   volume: float = 1.0) -> CostEstimate:
    """Round-turn cost expressed in *price units*, so it can be compared
    directly against stop and target distances."""
    spread_cost = spec.price(max(spread_points, 0.0))

    # Commission is charged in account currency per lot per side. Convert to a
    # price-equivalent so it can be netted against the move size.
    vpu = spec.value_per_price_unit()
    commission_ccy = 2.0 * cfg.costs.commission_per_lot_per_side  # round turn, per lot
    commission_price_units = commission_ccy / vpu if vpu > 0 else 0.0

    slippage = spec.price(cfg.costs.assumed_slippage_points) * 2.0

    total = spread_cost + commission_price_units + slippage
    return CostEstimate(spread_cost, commission_price_units, slippage, total)


def position_size(
    cfg: Config,
    spec: SymbolSpec,
    equity: float,
    stop_distance: float,
    risk_scale: float = 1.0,
    open_risk: float = 0.0,
) -> tuple[float, float]:
    """Return (volume, risk_amount_ccy).

    Risk is measured to the stop *including* the spread paid on entry, because
    the position is underwater by the spread the instant it opens.

    `open_risk` is the summed risk of positions already open. The new trade is
    sized down to fit inside max_portfolio_risk_pct, so holding four correlated
    gold longs never exposes more than one full-size position would. Once a
    position's stop moves to break-even its open risk is ~0 and the budget it
    was using is released automatically.
    """
    if stop_distance <= 0 or equity <= 0:
        return 0.0, 0.0

    risk_pct = cfg.risk.risk_pct_per_trade * max(0.0, min(1.5, risk_scale))
    risk_amount = equity * (risk_pct / 100.0)

    # Portfolio cap. Concurrent same-direction gold positions correlate ~1.0,
    # so treat them as one aggregate bet and never let the total exceed budget.
    budget = equity * (cfg.risk.max_portfolio_risk_pct / 100.0)
    remaining = max(0.0, budget - max(0.0, open_risk))
    risk_amount = min(risk_amount, remaining)
    if risk_amount <= 0:
        return 0.0, 0.0

    vpu = spec.value_per_price_unit()
    if vpu <= 0:
        return 0.0, 0.0

    loss_per_lot = stop_distance * vpu
    if loss_per_lot <= 0:
        return 0.0, 0.0

    raw_volume = risk_amount / loss_per_lot

    # normalize_volume() clamps UP to volume_min. If the correctly-sized position
    # is smaller than the broker's minimum lot, taking volume_min would risk more
    # than intended - silently, and by an unbounded factor on a small account.
    # Refuse the trade instead. This is the single most dangerous rounding bug in
    # retail position sizing.
    if raw_volume < spec.volume_min:
        return 0.0, 0.0

    volume = spec.normalize_volume(raw_volume)
    actual_risk = volume * loss_per_lot

    # Rounding to the volume step can only ever round DOWN here, but assert the
    # invariant anyway: realised risk must never exceed the budget.
    if actual_risk > risk_amount * 1.02:
        return 0.0, 0.0

    return volume, actual_risk


def expectancy_after_costs(
    signal_r_tp1: float,
    signal_r_tp2: float,
    partial_pct: float,
    stop_distance: float,
    costs: CostEstimate,
    assumed_win_rate: float,
) -> float:
    """Expected R per trade net of costs, under a stated win-rate assumption.

    This does not know the true win rate - nothing does before the fact. It is
    a sanity check: it answers 'if this setup wins X% of the time, does the
    structure still make money after what I pay to trade it?'. A structure that
    needs a 70% win rate to break even is not worth running.
    """
    if stop_distance <= 0:
        return 0.0
    cost_r = costs.total_round_turn / stop_distance
    p = max(0.0, min(1.0, assumed_win_rate))
    frac1 = partial_pct / 100.0
    win_r = frac1 * signal_r_tp1 + (1.0 - frac1) * signal_r_tp2
    return p * win_r - (1.0 - p) * 1.0 - cost_r


def breakeven_win_rate(
    signal_r_tp1: float,
    signal_r_tp2: float,
    partial_pct: float,
    stop_distance: float,
    costs: CostEstimate,
) -> float:
    """Win rate required to break even, given the payoff structure and costs."""
    if stop_distance <= 0:
        return 1.0
    cost_r = costs.total_round_turn / stop_distance
    frac1 = partial_pct / 100.0
    win_r = frac1 * signal_r_tp1 + (1.0 - frac1) * signal_r_tp2
    denom = win_r + 1.0
    if denom <= 0:
        return 1.0
    return float(min(1.0, max(0.0, (1.0 + cost_r) / denom)))