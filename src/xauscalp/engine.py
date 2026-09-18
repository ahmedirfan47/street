"""The engine.

One decision path, used identically by the backtester and the live runner. If
you find yourself writing `if backtest:` in here, stop - that divergence is
exactly what makes a backtest unreliable.

Per closed bar:
    1. Roll period counters, refresh spread statistics.
    2. Manage any open positions (partials, break-even, trail, time stops).
    3. If inside the pre-session lead window, build and store the SessionPlan.
    4. Run the guard stack. Any veto ends the bar.
    5. Generate candidate signals from the permitted setup families.
    6. Filter by session direction, quality floor and entry spacing.
    7. Cost-check the best candidate, size it against the portfolio budget, send.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from .analytics.indicators import enrich
from .analytics.levels import Level, build_levels
from .analytics.structure import find_swings
from .broker.base import Broker, Position
from .config import Config
from .execution.manager import TradeManager
from .journal.store import Journal
from .news.calendar import NewsFilter
from .risk.guards import GuardStack, RiskState
from .risk.sizing import SymbolSpec, estimate_costs, position_size
from .sessions import Session, SessionClock
from .strategy.base import Rejection, SessionPlan, Signal
from .strategy.daytrade import find_orb_signal, find_vwap_pullback_signal
from .strategy.displacement import find_displacement_signal
from .strategy.fibonacci import find_fib_signals
from .strategy.liquidity_sweep import find_sweep_signal
from .strategy.session_plan import build_session_plan

log = logging.getLogger(__name__)


@dataclass
class BarContext:
    now: datetime
    m5: pd.DataFrame           # enriched, closed bars only
    htf: pd.DataFrame
    spread_points: float
    equity: float
    balance: float
    positions: list[Position]


@dataclass
class EngineStats:
    bars_processed: int = 0
    plans_built: int = 0
    signals_generated: int = 0
    signals_taken: int = 0
    vetoes: dict[str, int] = field(default_factory=dict)

    def veto(self, reason: str) -> None:
        self.vetoes[reason] = self.vetoes.get(reason, 0) + 1


class Engine:
    def __init__(self, cfg: Config, broker: Broker, spec: SymbolSpec,
                 journal: Journal | None = None, start_equity: float | None = None):
        self.cfg = cfg
        self.broker = broker
        self.spec = spec
        self.clock = SessionClock(cfg.sessions)
        self.guards = GuardStack(cfg, self.clock)
        self.news = NewsFilter(cfg.news)
        self.manager = TradeManager(cfg, broker, spec)
        self.journal = journal
        self.stats = EngineStats()

        eq = start_equity if start_equity is not None else cfg.backtest.initial_equity
        self.state = RiskState(start_equity=eq, day_start_equity=eq,
                               week_start_equity=eq)

        self.plans: dict[str, SessionPlan] = {}     # keyed by "<day>:<session>"
        self._spread_hist: deque[float] = deque(maxlen=400)
        self._level_use: dict[str, int] = {}
        self._used_fvgs: set[str] = set()
        self._used_legs: dict[str, int] = {}
        self._used_orb: set[str] = set()
        self._setup_last_bar: dict[str, int] = {}
        self._levels: list[Level] = []
        self._levels_stamp: datetime | None = None
        self._bar_index = 0
        self._last_entry_bar = -10**9
        self._last_plan_key: str | None = None

    # ------------------------------------------------------------------ helpers
    def _plan_key(self, now: datetime, session: Session) -> str:
        return f"{self.clock.trading_day_key(now)}:{session.value}"

    def current_plan(self, now: datetime, session: Session) -> SessionPlan | None:
        return self.plans.get(self._plan_key(now, session))

    def median_spread(self) -> float:
        if not self._spread_hist:
            return self.cfg.backtest.spread_points_mean
        s = sorted(self._spread_hist)
        return s[len(s) // 2]

    def _refresh_levels(self, m5: pd.DataFrame, now: datetime) -> None:
        """Rebuild the level map at most once every 15 minutes - it is derived
        from completed sessions and does not change bar to bar."""
        if self._levels_stamp is not None and \
                (now - self._levels_stamp).total_seconds() < 900 and self._levels:
            return
        swings = find_swings(m5.tail(400), left=2, right=2)
        self._levels = build_levels(
            m5, self.clock, now,
            swing_highs=[s.price for s in swings if s.kind == "high"],
            swing_lows=[s.price for s in swings if s.kind == "low"],
        )
        self._levels_stamp = now

    # -------------------------------------------------------------------- plans
    def maybe_build_plan(self, ctx: BarContext) -> SessionPlan | None:
        due = self.clock.is_plan_time(ctx.now)
        if due is None:
            return None
        session, open_utc = due
        key = f"{self.clock.trading_day_key(open_utc)}:{session.value}"
        if key in self.plans:
            return None

        plan = build_session_plan(
            self.cfg, ctx.now, session, open_utc, ctx.m5, ctx.htf,
            ctx.spread_points, self.median_spread(),
        )
        self.plans[key] = plan
        self.stats.plans_built += 1
        self._last_plan_key = key
        if self.journal:
            self.journal.log_plan(plan)
        log.info(plan.render())
        return plan

    # --------------------------------------------------------------------- main
    def on_bar(self, ctx: BarContext) -> list[str]:
        actions: list[str] = []
        self.stats.bars_processed += 1
        self._bar_index += 1

        if ctx.spread_points > 0:
            self._spread_hist.append(ctx.spread_points)

        session = self.clock.active_session(ctx.now)
        self.state.roll_periods(ctx.now, ctx.equity, self.clock, session)

        atr_value = float(ctx.m5["atr"].iloc[-1]) if "atr" in ctx.m5 else 0.0
        price = float(ctx.m5["close"].iloc[-1])

        # ---- 1. manage open positions ---------------------------------------
        if ctx.positions:
            actions += self.manager.update(ctx.positions, price, atr_value, ctx.now)

        # Hard flatten conditions take precedence over everything else.
        if ctx.positions and self.clock.is_friday_flatten(ctx.now):
            actions += self.manager.flatten_all(ctx.positions, "friday_flatten")
            return actions

        blackout = self.news.blackout(ctx.now)
        if blackout and ctx.positions:
            actions += self.manager.flatten_all(ctx.positions, "news_blackout")
            return actions

        # ---- 2. pre-session planning ----------------------------------------
        self.maybe_build_plan(ctx)

        # ---- 3. can we trade at all? ----------------------------------------
        if session == Session.OFF:
            return actions

        if blackout:
            self.stats.veto("news_blackout")
            return actions

        acct_v = self.guards.check_account(self.state, ctx.equity, ctx.now)
        if not acct_v.allowed:
            self.stats.veto(acct_v.reason)
            if acct_v.reason.startswith(("equity_floor", "daily_loss", "weekly_loss")) \
                    and self.journal:
                self.journal.log_event(ctx.now, "WARNING", "halt", acct_v.reason,
                                       acct_v.detail)
            return actions

        plan = self.current_plan(ctx.now, session)
        if plan is None or not plan.tradable or not plan.allowed_setups:
            self.stats.veto("no_tradable_plan")
            return actions

        sess_v = self.guards.check_session(self.state, ctx.now, session,
                                           len(ctx.positions), plan.max_trades)
        if not sess_v.allowed:
            self.stats.veto(sess_v.reason)
            return actions

        # ---- 4. generate candidates -----------------------------------------
        self._refresh_levels(ctx.m5, ctx.now)
        if not self._levels:
            self.stats.veto("no_levels")
            return actions

        candidates: list[Signal] = []
        rejections: list[Rejection] = []

        # The smallest stop that can clear current round-turn costs at the
        # configured ratio. Derived live from the actual spread, so the system
        # automatically demands wider stops when conditions deteriorate rather
        # than generating setups that the cost guard will only reject later.
        cost_now = estimate_costs(self.cfg, self.spec, ctx.spread_points)
        min_viable_stop = 0.0
        if self.cfg.costs.max_cost_to_stop_ratio > 0:
            min_viable_stop = (cost_now.total_round_turn
                               / self.cfg.costs.max_cost_to_stop_ratio) * 1.02

        sig_a, rej_a = find_sweep_signal(
            self.cfg, ctx.m5, plan, session, self._levels, ctx.now,
            self._level_use, self._bar_index, min_viable_stop=min_viable_stop,
        )
        rejections += rej_a
        if sig_a:
            candidates.append(sig_a)

        sig_b, rej_b = find_displacement_signal(
            self.cfg, ctx.m5, plan, session, self._levels, ctx.now, self._used_fvgs,
            min_viable_stop=min_viable_stop,
        )
        rejections += rej_b
        if sig_b:
            candidates.append(sig_b)

        sigs_c, rej_c = find_fib_signals(
            self.cfg, ctx.m5, plan, session, self._levels, ctx.now,
            self._used_legs, self._bar_index, min_viable_stop=min_viable_stop,
        )
        rejections += rej_c
        candidates.extend(sigs_c)

        bounds = self.clock.current_bounds(ctx.now, session)

        sig_d, rej_d = find_orb_signal(
            self.cfg, ctx.m5, plan, session, bounds, self._levels, ctx.now,
            self._used_orb, min_viable_stop=min_viable_stop,
        )
        rejections += rej_d
        if sig_d:
            candidates.append(sig_d)

        sig_e, rej_e = find_vwap_pullback_signal(
            self.cfg, ctx.m5, plan, session, bounds, self._levels, ctx.now,
            self._setup_last_bar, self._bar_index, min_viable_stop=min_viable_stop,
        )
        rejections += rej_e
        if sig_e:
            candidates.append(sig_e)

        if self.journal and rejections:
            self.journal.log_rejections(rejections)

        if not candidates:
            return actions

        self.stats.signals_generated += len(candidates)

        # ---- DIRECTIONAL LOCK ------------------------------------------------
        # The session committed to a side before it opened. Anything on the
        # other side is refused, however good it looks. This is the whole point
        # of deciding direction first: no hedging, no mid-session flip-flopping.
        if plan.direction in ("long", "short"):
            kept = [c for c in candidates if c.direction == plan.direction]
            if len(kept) < len(candidates):
                self.stats.veto(f"against_session_direction_{plan.direction}")
            candidates = kept
            if not candidates:
                return actions

        # ---- QUALITY FLOOR ---------------------------------------------------
        # Trade-count limits are ceilings, not targets. The bot does not fill a
        # quota with weak setups just because slots remain.
        floor = self.cfg.risk.min_signal_quality
        graded = [c for c in candidates if c.quality >= floor]
        if not graded:
            self.stats.veto("below_min_quality")
            return actions
        candidates = graded

        # ---- ENTRY SPACING ---------------------------------------------------
        # Stops a burst of signals off one impulse from all piling into the same
        # move, which would be one position in four pieces.
        gap = self._bar_index - self._last_entry_bar
        if gap < self.cfg.sessions.min_bars_between_entries:
            self.stats.veto("too_soon_after_last_entry")
            return actions

        best = max(candidates, key=lambda s: s.quality)

        # ---- 5. cost and viability check -------------------------------------
        verdict, costs = self.guards.check_signal(
            best, self.spec, ctx.spread_points, self.median_spread(), ctx.equity,
        )
        if not verdict.allowed:
            self.stats.veto(verdict.reason)
            if self.journal:
                self.journal.log_signal(best, accepted=False,
                                        veto_reason=verdict.reason,
                                        spread_points=ctx.spread_points)
            return actions

        # ---- 6. size and send -------------------------------------------------
        # Fibonacci profiles may carry their own risk weight, so an experimental
        # profile can be run at reduced size while it is being evaluated.
        risk_scale = plan.risk_scale * float(best.meta.get("risk_weight", 1.0))

        # Summed risk of everything already open. A position whose stop has
        # moved to break-even contributes ~0 and frees its budget back.
        vpu = self.spec.value_per_price_unit()
        open_risk = 0.0
        for p in ctx.positions:
            if p.direction == "long":
                at_risk = max(0.0, p.entry_price - p.stop)
            else:
                at_risk = max(0.0, p.stop - p.entry_price)
            open_risk += at_risk * vpu * p.volume

        volume, risk_amount = position_size(
            self.cfg, self.spec, ctx.equity, best.stop_distance, risk_scale,
            open_risk=open_risk,
        )
        if volume <= 0:
            self.stats.veto("volume_below_minimum_or_portfolio_cap")
            if self.journal:
                self.journal.log_signal(best, accepted=False,
                                        veto_reason="volume_below_minimum",
                                        spread_points=ctx.spread_points)
            return actions

        res = self.broker.market_order(
            self.cfg.symbol.name, best.direction, volume,
            self.spec.normalize_price(best.stop),
            self.spec.normalize_price(best.tp2),
            comment=f"{best.setup[:8]}_{session.value[:3]}",
        )

        if not res.ok:
            log.warning("Order rejected: %s", res.error)
            self.stats.veto(f"order_failed:{res.error[:40]}")
            if self.journal:
                self.journal.log_signal(best, accepted=False,
                                        veto_reason=f"order_failed:{res.error}",
                                        spread_points=ctx.spread_points)
            return actions

        self.stats.signals_taken += 1
        self.state.register_open()
        self._last_entry_bar = self._bar_index

        if best.setup == "sweep" and best.meta.get("level_key"):
            self._level_use[best.meta["level_key"]] = self._bar_index
        if best.setup == "displacement" and best.meta.get("fvg_key"):
            self._used_fvgs.add(best.meta["fvg_key"])
        if best.setup.startswith("fib") and best.meta.get("leg_key"):
            self._used_legs[best.meta["leg_key"]] = self._bar_index
        if best.setup == "orb" and best.meta.get("orb_key"):
            self._used_orb.add(best.meta["orb_key"])
        self._setup_last_bar[best.setup] = self._bar_index

        for p in self.broker.positions(self.cfg.symbol.name):
            if p.ticket == res.ticket:
                meta = {"setup": best.setup, "session": session.value,
                        "quality": best.quality, "level": best.level_name}
                # Carry the Fibonacci profile through to the closed trade so the
                # two profiles can be compared directly in the journal.
                if best.meta.get("profile"):
                    meta["profile"] = best.meta["profile"]
                    meta["retracement_depth"] = best.meta.get("retracement_depth")
                    meta["leg_length"] = best.meta.get("leg_length")
                p.meta.update(meta)
                self.manager.register(p, meta)
                break

        if self.journal:
            self.journal.log_signal(
                best, accepted=True,
                breakeven_wr=verdict.detail.get("breakeven_win_rate"),
                cost_to_stop=verdict.detail.get("cost_to_stop"),
                spread_points=ctx.spread_points,
            )

        actions.append(
            f"OPEN {best.direction} {volume} @{res.price:.2f} "
            f"SL {best.stop:.2f} TP {best.tp2:.2f} "
            f"risk ${risk_amount:.2f} (open ${open_risk:.0f}) "
            f"| {best.setup} | {best.reason}"
        )
        log.info(actions[-1])
        return actions

    # ---------------------------------------------------------------- utilities
    def prepare_frames(self, m5_raw: pd.DataFrame,
                       htf_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        r = self.cfg.regime
        return (enrich(m5_raw, r.atr_period, r.adx_period, r.atr_lookback_bars),
                enrich(htf_raw, r.atr_period, r.adx_period, r.atr_lookback_bars))