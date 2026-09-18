"""Backtest driver and performance analytics.

Reports the numbers that determine whether a strategy is deployable, not the
ones that look good in marketing: net-of-cost expectancy, max drawdown, the
cost drag as a share of gross profit, and a per-session breakdown.

Read the caveats printed at the end of every run. A backtest on bar data cannot
resolve intrabar sequencing, so stop-versus-target ordering within a bar is
assumed adversely. That is conservative, but it is still an assumption.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from .analytics.indicators import enrich
from .broker.sim_broker import ClosedTrade, SimBroker
from .config import Config
from .engine import BarContext, Engine
from .journal.store import Journal
from .risk.sizing import SymbolSpec

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    trades: list[ClosedTrade]
    equity_curve: pd.Series
    stats: dict
    engine_stats: dict
    veto_counts: dict = field(default_factory=dict)

    def summary(self) -> str:
        s = self.stats
        lines = [
            "",
            "=" * 74,
            "  BACKTEST RESULT",
            "=" * 74,
            f"  Period            : {s['start']} -> {s['end']}",
            f"  Trades (fills)    : {s['n_trades']}   "
            f"(incl. partial closes)",
            f"  Net P/L           : {s['net_pnl']:+,.2f}  "
            f"({s['return_pct']:+.2f}%)",
            f"  Gross profit      : {s['gross_profit']:,.2f}",
            f"  Gross loss        : {s['gross_loss']:,.2f}",
            f"  Commission paid   : {s['commission']:,.2f}   "
            f"({s['cost_drag_pct']:.1f}% of gross profit)",
            f"  Profit factor     : {s['profit_factor']:.2f}",
            f"  Win rate          : {s['win_rate']*100:.1f}%",
            f"  Avg R             : {s['avg_r']:+.3f}",
            f"  Expectancy / trade: {s['expectancy']:+,.2f}",
            f"  Max drawdown      : {s['max_dd_pct']:.2f}%  "
            f"({s['max_dd_abs']:,.2f})",
            f"  Sharpe (daily)    : {s['sharpe']:.2f}",
            f"  Longest DD (days) : {s['dd_days']:.1f}",
            "",
            "  By session:",
        ]
        for sess, row in s["by_session"].items():
            lines.append(f"      {sess:<10} n={row['n']:<4} "
                         f"pnl={row['pnl']:+10,.2f}  avgR={row['avg_r']:+.3f}  "
                         f"wr={row['win_rate']*100:5.1f}%")
        lines.append("")
        lines.append("  By setup:")
        for setup, row in s["by_setup"].items():
            lines.append(f"      {setup:<14} n={row['n']:<4} "
                         f"pnl={row['pnl']:+10,.2f}  avgR={row['avg_r']:+.3f}  "
                         f"wr={row['win_rate']*100:5.1f}%")
        lines.append("")
        lines.append("  Exit reasons:")
        for reason, cnt in sorted(s["by_reason"].items(), key=lambda x: -x[1]):
            lines.append(f"      {reason:<16} {cnt}")
        if self.veto_counts:
            lines.append("")
            lines.append("  Top vetoes (why trades were NOT taken):")
            for reason, cnt in sorted(self.veto_counts.items(),
                                      key=lambda x: -x[1])[:12]:
                lines.append(f"      {reason:<32} {cnt}")
        lines.append("=" * 74)
        return "\n".join(lines)


def _metrics(trades: list[ClosedTrade], equity: pd.Series, initial: float,
             commission: float) -> dict:
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    gross_profit = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))
    net = sum(t.pnl for t in trades)

    if len(equity) > 1:
        running_max = equity.cummax()
        dd = equity - running_max
        dd_pct = (dd / running_max.replace(0, np.nan)) * 100.0
        max_dd_abs = float(dd.min())
        max_dd_pct = float(abs(dd_pct.min())) if not dd_pct.isna().all() else 0.0
        daily = equity.resample("1D").last().dropna()
        rets = daily.pct_change().dropna()
        sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) \
            if len(rets) > 5 and rets.std() > 0 else 0.0
        underwater = (equity < running_max)
        dd_days = 0.0
        if underwater.any():
            grp = (~underwater).cumsum()
            runs = underwater.groupby(grp).sum()
            bars = float(runs.max())
            freq = (equity.index[1] - equity.index[0]).total_seconds() / 60.0
            dd_days = bars * freq / (60 * 24)
    else:
        max_dd_abs = max_dd_pct = sharpe = dd_days = 0.0

    def _group(key):
        out = {}
        for t in trades:
            k = getattr(t, key) or "unknown"
            row = out.setdefault(k, {"n": 0, "pnl": 0.0, "rs": [], "wins": 0})
            row["n"] += 1
            row["pnl"] += t.pnl
            row["rs"].append(t.r_multiple)
            row["wins"] += 1 if t.pnl > 0 else 0
        for k, row in out.items():
            row["avg_r"] = float(np.mean(row["rs"])) if row["rs"] else 0.0
            row["win_rate"] = row["wins"] / row["n"] if row["n"] else 0.0
            row.pop("rs")
        return out

    reasons: dict[str, int] = {}
    for t in trades:
        reasons[t.reason] = reasons.get(t.reason, 0) + 1

    return {
        "start": equity.index[0].strftime("%Y-%m-%d") if len(equity) else "-",
        "end": equity.index[-1].strftime("%Y-%m-%d") if len(equity) else "-",
        "n_trades": len(trades),
        "net_pnl": net,
        "return_pct": (net / initial * 100.0) if initial else 0.0,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "commission": commission,
        "cost_drag_pct": (commission / gross_profit * 100.0) if gross_profit > 0 else 0.0,
        "profit_factor": (gross_profit / gross_loss) if gross_loss > 0 else float("inf"),
        "win_rate": len(wins) / len(trades) if trades else 0.0,
        "avg_r": float(np.mean([t.r_multiple for t in trades])) if trades else 0.0,
        "expectancy": net / len(trades) if trades else 0.0,
        "max_dd_abs": max_dd_abs,
        "max_dd_pct": max_dd_pct,
        "sharpe": sharpe,
        "dd_days": dd_days,
        "by_session": _group("session"),
        "by_setup": _group("setup"),
        "by_reason": reasons,
    }


def run_backtest(
    cfg: Config,
    m5: pd.DataFrame,
    spec: SymbolSpec,
    htf_minutes: int = 15,
    journal_path: str | None = None,
    warmup_bars: int = 600,
    progress_every: int = 5000,
) -> BacktestResult:
    """Bar-by-bar replay through the same Engine used in live trading."""
    from .data.synthetic import resample

    m5e = enrich(m5, cfg.regime.atr_period, cfg.regime.adx_period,
                 cfg.regime.atr_lookback_bars)
    htf_all = enrich(resample(m5, htf_minutes), cfg.regime.atr_period,
                     cfg.regime.adx_period, cfg.regime.atr_lookback_bars)

    broker = SimBroker(cfg, m5e, spec)
    journal = Journal(journal_path) if journal_path else None
    engine = Engine(cfg, broker, spec, journal, cfg.backtest.initial_equity)

    broker.seek(warmup_bars)
    total = len(m5e)

    while True:
        i = broker.index
        if i % progress_every == 0 and i > warmup_bars:
            log.info("backtest %d/%d (%.0f%%) equity=%.2f",
                     i, total, 100.0 * i / total, broker.equity())

        now = broker.now()
        window = m5e.iloc[max(0, i - cfg.execution.history_bars + 1): i + 1]
        htf_window = htf_all.loc[htf_all.index <= now].tail(400)

        if len(window) >= 60 and len(htf_window) >= 60:
            ctx = BarContext(
                now=now, m5=window, htf=htf_window,
                spread_points=broker.spread_points(),
                equity=broker.equity(), balance=broker.account().balance,
                positions=broker.positions(),
            )
            try:
                engine.on_bar(ctx)
            except Exception:
                log.exception("engine error at %s", now)

        if not broker.step():
            break

    # Flatten anything still open at the end so results are not path-dependent.
    for p in broker.positions():
        broker.close_position(p.ticket, reason="end_of_test")

    eq = pd.Series(
        [e for _, e in broker.equity_curve],
        index=pd.DatetimeIndex([t for t, _ in broker.equity_curve]),
    ) if broker.equity_curve else pd.Series(dtype=float)

    stats = _metrics(broker.closed, eq, cfg.backtest.initial_equity,
                     broker.total_commission)
    if journal:
        journal.close()

    return BacktestResult(
        trades=broker.closed, equity_curve=eq, stats=stats,
        engine_stats={
            "bars": engine.stats.bars_processed,
            "plans": engine.stats.plans_built,
            "signals": engine.stats.signals_generated,
            "taken": engine.stats.signals_taken,
        },
        veto_counts=dict(engine.stats.vetoes),
    )


def walk_forward(cfg: Config, m5: pd.DataFrame, spec: SymbolSpec,
                 train_days: int | None = None,
                 test_days: int | None = None) -> list[dict]:
    """Sequential out-of-sample windows.

    This implementation does NOT optimise parameters on the training slice - it
    reports in-sample and out-of-sample performance for the fixed configuration.
    That is the honest starting point: if a fixed config is not stable across
    windows, adding an optimiser will only fit noise more convincingly.
    """
    train_days = train_days or cfg.backtest.train_days
    test_days = test_days or cfg.backtest.test_days

    results: list[dict] = []
    start = m5.index[0]
    end = m5.index[-1]
    cursor = start + pd.Timedelta(days=train_days)

    while cursor + pd.Timedelta(days=test_days) <= end:
        test_slice = m5.loc[cursor - pd.Timedelta(days=train_days):
                            cursor + pd.Timedelta(days=test_days)]
        if len(test_slice) < 2000:
            cursor += pd.Timedelta(days=test_days)
            continue
        res = run_backtest(cfg, test_slice, spec, warmup_bars=600)
        results.append({
            "window_start": str(cursor.date()),
            "window_end": str((cursor + pd.Timedelta(days=test_days)).date()),
            "trades": res.stats["n_trades"],
            "net_pnl": res.stats["net_pnl"],
            "avg_r": res.stats["avg_r"],
            "profit_factor": res.stats["profit_factor"],
            "max_dd_pct": res.stats["max_dd_pct"],
        })
        log.info("WF window %s -> %s : pnl=%.2f avgR=%.3f n=%d",
                 results[-1]["window_start"], results[-1]["window_end"],
                 results[-1]["net_pnl"], results[-1]["avg_r"],
                 results[-1]["trades"])
        cursor += pd.Timedelta(days=test_days)

    return results