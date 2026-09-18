"""Risk guards.

Every guard here can veto a trade. They run *after* the strategy produces a
signal and *before* anything reaches the broker. The ordering is deliberate:
cheap state checks first, market-dependent checks last.

The most important one is the cost guard. On XAUUSD a round-turn of spread plus
commission plus realistic slippage is commonly 20-35 points ($0.20-$0.35). A
setup risking $2.00 is paying 10-17% of its risk in friction before it starts.
Anything tighter than that is arithmetically hopeless, however good the chart
looks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ..config import Config
from ..sessions import Session, SessionClock
from ..strategy.base import Signal
from .sizing import CostEstimate, SymbolSpec, breakeven_win_rate, estimate_costs


@dataclass
class GuardVerdict:
    allowed: bool
    reason: str = ""
    detail: dict = field(default_factory=dict)


@dataclass
class RiskState:
    """Mutable trading state tracked across the session/day/week."""
    start_equity: float
    day_key: str = ""
    week_key: str = ""
    day_start_equity: float = 0.0
    week_start_equity: float = 0.0
    trades_today: int = 0
    trades_this_session: int = 0
    consecutive_losses: int = 0
    cooloff_until: datetime | None = None
    session_key: str = ""
    halted: bool = False
    halt_reason: str = ""

    def roll_periods(self, now: datetime, equity: float, clock: SessionClock,
                     session: Session) -> None:
        dk = clock.trading_day_key(now)
        wk = f"{now.isocalendar().year}-W{now.isocalendar().week:02d}"
        sk = f"{dk}:{session.value}"
        if dk != self.day_key:
            self.day_key = dk
            self.day_start_equity = equity
            self.trades_today = 0
            # A new day clears a daily-loss halt but not an equity-floor halt.
            if self.halt_reason.startswith("daily"):
                self.halted = False
                self.halt_reason = ""
        if wk != self.week_key:
            self.week_key = wk
            self.week_start_equity = equity
            if self.halt_reason.startswith("weekly"):
                self.halted = False
                self.halt_reason = ""
        if sk != self.session_key:
            self.session_key = sk
            self.trades_this_session = 0

    def register_close(self, pnl: float, now: datetime, cfg: Config) -> None:
        if pnl < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= cfg.risk.consecutive_loss_cooloff:
                self.cooloff_until = now + timedelta(minutes=cfg.risk.cooloff_minutes)
                self.consecutive_losses = 0
        else:
            self.consecutive_losses = 0

    def register_open(self) -> None:
        self.trades_today += 1
        self.trades_this_session += 1


class GuardStack:
    def __init__(self, cfg: Config, clock: SessionClock):
        self.cfg = cfg
        self.clock = clock

    # -------------------------------------------------------------- portfolio
    def check_account(self, state: RiskState, equity: float, now: datetime) -> GuardVerdict:
        c = self.cfg.risk

        if state.halted:
            return GuardVerdict(False, f"halted:{state.halt_reason}")

        floor = state.start_equity * (c.equity_floor_pct_of_start / 100.0)
        if equity < floor:
            state.halted = True
            state.halt_reason = "equity_floor"
            return GuardVerdict(False, "equity_floor",
                                {"equity": equity, "floor": floor})

        if state.day_start_equity > 0:
            day_pnl_pct = (equity - state.day_start_equity) / state.day_start_equity * 100.0
            if day_pnl_pct <= -c.max_daily_loss_pct:
                state.halted = True
                state.halt_reason = "daily_loss_limit"
                return GuardVerdict(False, "daily_loss_limit", {"day_pnl_pct": day_pnl_pct})
            if c.stop_on_daily_profit and day_pnl_pct >= c.max_daily_profit_pct:
                state.halted = True
                state.halt_reason = "daily_profit_target"
                return GuardVerdict(False, "daily_profit_target", {"day_pnl_pct": day_pnl_pct})

        if state.week_start_equity > 0:
            wk_pnl_pct = (equity - state.week_start_equity) / state.week_start_equity * 100.0
            if wk_pnl_pct <= -c.max_weekly_loss_pct:
                state.halted = True
                state.halt_reason = "weekly_loss_limit"
                return GuardVerdict(False, "weekly_loss_limit", {"week_pnl_pct": wk_pnl_pct})

        if state.cooloff_until and now < state.cooloff_until:
            return GuardVerdict(False, "cooloff_active",
                                {"until": state.cooloff_until.isoformat()})

        if state.trades_today >= c.max_trades_per_day:
            return GuardVerdict(False, "max_trades_per_day")

        return GuardVerdict(True)

    # ---------------------------------------------------------------- session
    def check_session(self, state: RiskState, now: datetime, session: Session,
                      open_positions: int, plan_max_trades: int) -> GuardVerdict:
        c = self.cfg.risk

        if session == Session.OFF:
            return GuardVerdict(False, "outside_session")

        if open_positions >= c.max_concurrent_positions:
            return GuardVerdict(False, "max_concurrent_positions")

        if state.trades_this_session >= min(c.max_trades_per_session, plan_max_trades):
            return GuardVerdict(False, "max_trades_per_session")

        mins_left = self.clock.minutes_to_session_end(now, session)
        if mins_left < self.cfg.sessions.no_new_trades_last_minutes:
            return GuardVerdict(False, "too_close_to_session_end",
                                {"minutes_left": round(mins_left, 1)})

        if self.clock.is_friday_flatten(now):
            return GuardVerdict(False, "friday_flatten_window")

        return GuardVerdict(True)

    # ------------------------------------------------------------------ trade
    def check_signal(
        self,
        signal: Signal,
        spec: SymbolSpec,
        spread_points: float,
        median_spread_points: float,
        equity: float,
    ) -> tuple[GuardVerdict, CostEstimate | None]:
        c = self.cfg.costs

        if spread_points > c.max_spread_points:
            return GuardVerdict(False, "spread_too_wide",
                                {"spread": spread_points,
                                 "max": c.max_spread_points}), None

        # Sudden spread expansion is the market telling you something is
        # happening that your model has not priced.
        if median_spread_points > 0 and \
                spread_points > median_spread_points * self.cfg.news.spread_spike_multiple:
            return GuardVerdict(False, "spread_spike",
                                {"spread": spread_points,
                                 "median": median_spread_points}), None

        costs = estimate_costs(self.cfg, spec, spread_points)
        stop_dist = signal.stop_distance
        tp1_dist = abs(signal.tp1 - signal.entry)

        if stop_dist <= 0:
            return GuardVerdict(False, "invalid_stop"), costs

        cost_to_stop = costs.total_round_turn / stop_dist
        if cost_to_stop > c.max_cost_to_stop_ratio:
            return GuardVerdict(False, "cost_to_stop_ratio",
                                {"ratio": round(cost_to_stop, 4),
                                 "max": c.max_cost_to_stop_ratio,
                                 "cost_price_units": round(costs.total_round_turn, 3),
                                 "stop_distance": round(stop_dist, 3)}), costs

        if tp1_dist > 0:
            cost_to_target = costs.total_round_turn / tp1_dist
            if cost_to_target > c.max_cost_to_target_ratio:
                return GuardVerdict(False, "cost_to_target_ratio",
                                    {"ratio": round(cost_to_target, 4),
                                     "max": c.max_cost_to_target_ratio}), costs

        # Broker minimum stop distance.
        if spec.stops_level_points > 0:
            min_dist = spec.price(spec.stops_level_points)
            if stop_dist < min_dist or tp1_dist < min_dist:
                return GuardVerdict(False, "inside_broker_stops_level",
                                    {"stops_level_points": spec.stops_level_points}), costs

        # Structural viability check.
        be = breakeven_win_rate(
            signal.r_to_tp1, signal.r_to_tp2,
            self.cfg.manage.partial_at_tp1_pct, stop_dist, costs,
        )
        if be > 0.62:
            return GuardVerdict(False, "breakeven_winrate_too_high",
                                {"breakeven_win_rate": round(be, 3)}), costs

        return GuardVerdict(True, "", {"breakeven_win_rate": round(be, 3),
                                       "cost_to_stop": round(cost_to_stop, 4)}), costs