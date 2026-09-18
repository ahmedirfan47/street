#!/usr/bin/env python
"""Entrypoint: backtesting.

    python backtest.py --source mt5            # real MT5 history (recommended)
    python backtest.py --source synthetic      # offline smoke test only
    python backtest.py --source mt5 --walk-forward
    python backtest.py --source mt5 --export data/xau_m5.csv
    python backtest.py --source csv --csv data/xau_m5.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from xauscalp.backtester import run_backtest, walk_forward   # noqa: E402
from xauscalp.config import load_config                      # noqa: E402
from xauscalp.logging_setup import setup_logging             # noqa: E402
from xauscalp.risk.sizing import SymbolSpec                  # noqa: E402


def fallback_spec(cfg) -> SymbolSpec:
    s = cfg.symbol
    return SymbolSpec(
        name=s.name, point=s.fallback_point, digits=s.fallback_digits,
        contract_size=s.fallback_contract_size, volume_min=s.fallback_volume_min,
        volume_max=s.fallback_volume_max, volume_step=s.fallback_volume_step,
        tick_value=s.fallback_contract_size * s.fallback_point,
        tick_size=s.fallback_point,
        stops_level_points=s.fallback_stops_level_points,
    )


def load_mt5_history(cfg, bars: int, terminal=None):
    from xauscalp.broker.mt5_broker import MT5Broker
    b = MT5Broker(cfg, terminal_path=terminal)
    if not b.connect():
        raise SystemExit("MT5 connection failed. Start the terminal and log in.")
    df = b.bars(cfg.symbol.name, cfg.execution.bar_timeframe, bars)
    spec = b.symbol_spec(cfg.symbol.name)
    b.disconnect()
    print(f"Loaded {len(df):,} {cfg.execution.bar_timeframe} bars "
          f"({df.index[0]} -> {df.index[-1]})")
    return df, spec


def main() -> int:
    ap = argparse.ArgumentParser(description="Backtest the XAUUSD scalper")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--source", choices=["mt5", "synthetic", "csv"], default="mt5")
    ap.add_argument("--bars", type=int, default=120000,
                    help="M5 bars to pull from MT5 (120k ~ 15 months)")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--export", default=None, help="Save fetched bars to CSV")
    ap.add_argument("--terminal", default=None)
    ap.add_argument("--walk-forward", action="store_true")
    ap.add_argument("--journal", default=None,
                    help="Write a decision journal, e.g. logs/backtest.sqlite")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.log_path, cfg.log_level)

    if args.source == "mt5":
        m5, spec = load_mt5_history(cfg, args.bars, args.terminal)
    elif args.source == "csv":
        if not args.csv:
            raise SystemExit("--csv path required with --source csv")
        m5 = pd.read_csv(args.csv, parse_dates=["time"], index_col="time")
        if m5.index.tz is None:
            m5.index = m5.index.tz_localize("UTC")
        spec = fallback_spec(cfg)
        print(f"Loaded {len(m5):,} bars from {args.csv}")
    else:
        from xauscalp.data.synthetic import generate_bars
        print("WARNING: synthetic data validates the CODE, not the STRATEGY.")
        m5 = generate_bars(cfg.backtest.start, cfg.backtest.end, seed=args.seed)
        spec = fallback_spec(cfg)
        print(f"Generated {len(m5):,} synthetic bars")

    if args.export:
        Path(args.export).parent.mkdir(parents=True, exist_ok=True)
        m5.to_csv(args.export)
        print(f"Exported bars to {args.export}")

    if args.walk_forward:
        rows = walk_forward(cfg, m5, spec)
        if not rows:
            print("Not enough data for walk-forward windows.")
            return 0
        df = pd.DataFrame(rows)
        print("\nWALK-FORWARD WINDOWS")
        print(df.to_string(index=False))
        pos = (df["net_pnl"] > 0).sum()
        print(f"\nProfitable windows: {pos}/{len(df)} "
              f"({pos/len(df)*100:.0f}%)")
        print(f"Median window P/L : {df['net_pnl'].median():+,.2f}")
        print("\nConsistency across windows matters far more than the total. "
              "A strategy carried by one or two windows is not a strategy.")
        return 0

    res = run_backtest(cfg, m5, spec, journal_path=args.journal)
    print(res.summary())
    print(f"\nEngine: {res.engine_stats}")
    print("\nCAVEATS")
    print("  - Bar data cannot resolve intrabar order; stop-before-target is assumed.")
    print("  - Modelled spread/slippage are estimates. Verify against your broker.")
    print("  - No account for swap on held positions or weekend gaps.")
    print("  - Past behaviour on any dataset is not a forecast.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())