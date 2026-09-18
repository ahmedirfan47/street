"""Compare setup combinations on the SAME data, in one run.

The full-system backtest told you the portfolio loses money. It did not tell you
which component is responsible, because the components compete for a shared
trade budget - a setup that fires often can starve a better one that fires
rarely, and then both look mediocre.

This isolates them. Each variant runs the identical engine over the identical
bars with only the enabled setups changed.

    python experiment.py --csv data/xau_m5.csv

If you have not exported the bars yet:

    python backtest.py --source mt5 --bars 120000 --export data/xau_m5.csv

Read NET P/L and PROFIT FACTOR. Ignore avg R and win rate in the standard
backtest summary - both are computed per fill, and a winning trade that takes a
partial books two fills while a loser books one, which biases them upward. The
position-level numbers printed here are the honest ones.
"""
from __future__ import annotations

import argparse
import copy
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from xauscalp.backtester import run_backtest          # noqa: E402
from xauscalp.config import load_config               # noqa: E402
from xauscalp.risk.sizing import SymbolSpec           # noqa: E402


def spec_from_cfg(cfg) -> SymbolSpec:
    s = cfg.symbol
    return SymbolSpec(
        name=s.name, point=s.fallback_point, digits=s.fallback_digits,
        contract_size=s.fallback_contract_size, volume_min=s.fallback_volume_min,
        volume_max=s.fallback_volume_max, volume_step=s.fallback_volume_step,
        tick_value=s.fallback_contract_size * s.fallback_point,
        tick_size=s.fallback_point,
        stops_level_points=s.fallback_stops_level_points,
    )


def variant(base, *, sweep=True, displacement=True,
            fib_recommended=True, fib_user=True):
    c = copy.deepcopy(base)
    c.sweep.enabled = sweep
    c.displacement.enabled = displacement
    for p in c.fib.profiles:
        if p.name == "recommended":
            p.enabled = fib_recommended
        elif p.name == "user":
            p.enabled = fib_user
    c.fib.enabled = fib_recommended or fib_user
    return c


def position_stats(res) -> dict:
    """Per-POSITION stats, not per-fill.

    A position that takes a partial produces two ClosedTrade records. Counting
    those as two outcomes inflates win rate and average R. Group by ticket.
    """
    by_ticket: dict[int, list] = {}
    for t in res.trades:
        by_ticket.setdefault(t.ticket, []).append(t)

    pnls = [sum(x.pnl for x in fills) for fills in by_ticket.values()]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gp = sum(wins)
    gl = abs(sum(losses))
    return {
        "positions": len(pnls),
        "win_rate": len(wins) / len(pnls) if pnls else 0.0,
        "profit_factor": (gp / gl) if gl > 0 else float("inf"),
        "avg_win": (gp / len(wins)) if wins else 0.0,
        "avg_loss": (gl / len(losses)) if losses else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/xau_m5.csv")
    ap.add_argument("--config", default="config/default.yaml")
    args = ap.parse_args()

    logging.basicConfig(level=logging.ERROR)   # suppress per-session plan spam

    path = Path(args.csv)
    if not path.exists():
        print(f"No bars at {path}.\nExport them first:\n"
              f"  python backtest.py --source mt5 --bars 120000 --export {path}")
        return 1

    m5 = pd.read_csv(path, parse_dates=["time"], index_col="time")
    if m5.index.tz is None:
        m5.index = m5.index.tz_localize("UTC")

    base = load_config(args.config)
    base.news.enabled = False      # same conditions for every variant
    spec = spec_from_cfg(base)

    print(f"\n{len(m5):,} bars  {m5.index[0].date()} -> {m5.index[-1].date()}")
    print(f"start equity {base.backtest.initial_equity:,.0f}\n")

    variants = {
        "sweep only": variant(base, displacement=False,
                              fib_recommended=False, fib_user=False),
        "sweep + displacement": variant(base, fib_recommended=False, fib_user=False),
        "sweep + fib_recommended": variant(base, displacement=False, fib_user=False),
        "fib_recommended only": variant(base, sweep=False, displacement=False,
                                        fib_user=False),
        "fib_user only": variant(base, sweep=False, displacement=False,
                                 fib_recommended=False),
        "everything (current)": variant(base),
    }

    rows = []
    for name, cfg in variants.items():
        print(f"running: {name} ...", flush=True)
        res = run_backtest(cfg, m5, spec, warmup_bars=600, progress_every=10**9)
        ps = position_stats(res)
        rows.append({
            "variant": name,
            "pos": ps["positions"],
            "net_pnl": round(res.stats["net_pnl"], 2),
            "ret_%": round(res.stats["return_pct"], 2),
            "PF": round(ps["profit_factor"], 2),
            "win_%": round(ps["win_rate"] * 100, 1),
            "avg_win": round(ps["avg_win"], 2),
            "avg_loss": round(ps["avg_loss"], 2),
            "maxDD_%": round(res.stats["max_dd_pct"], 2),
        })

    df = pd.DataFrame(rows).sort_values("net_pnl", ascending=False)
    print("\n" + "=" * 88)
    print("  PER-POSITION RESULTS  (partials merged - these are the honest numbers)")
    print("=" * 88)
    print(df.to_string(index=False))
    print("=" * 88)

    best = df.iloc[0]
    print(f"\nBest: {best['variant']}  ->  {best['net_pnl']:+,.2f} "
          f"({best['ret_%']:+.2f}%), PF {best['PF']:.2f} on {best['pos']} positions")
    print("\nA profit factor below 1.0 loses money. Above 1.2 on a few hundred")
    print("positions is worth forward-testing. Between the two is noise, and")
    print("small samples on any single period prove nothing either way - confirm")
    print("anything promising with:")
    print("  python backtest.py --source csv --csv "
          f"{args.csv} --walk-forward")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())