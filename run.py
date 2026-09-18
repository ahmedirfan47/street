#!/usr/bin/env python
"""Entrypoint: connect to MT5 and run the engine (paper by default).

    python run.py                 # paper mode, uses config/default.yaml
    python run.py --mode live     # live (still requires the real-account switch)
    python run.py --check         # connectivity + symbol spec check only
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from xauscalp.config import load_config          # noqa: E402
from xauscalp.logging_setup import setup_logging  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="XAUUSD session scalper")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--mode", choices=["paper", "live"], default=None)
    ap.add_argument("--login", type=int, default=None)
    ap.add_argument("--password", default=None)
    ap.add_argument("--server", default=None)
    ap.add_argument("--terminal", default=None, help="Path to terminal64.exe")
    ap.add_argument("--check", action="store_true",
                    help="Connect, print account and symbol spec, then exit")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.mode:
        cfg.execution.mode = args.mode
    setup_logging(cfg.log_path, cfg.log_level)

    login = args.login or (int(os.environ["MT5_LOGIN"])
                           if os.environ.get("MT5_LOGIN") else None)
    password = args.password or os.environ.get("MT5_PASSWORD")
    server = args.server or os.environ.get("MT5_SERVER")
    terminal = args.terminal or os.environ.get("MT5_TERMINAL_PATH")

    if args.check:
        from xauscalp.broker.mt5_broker import MT5Broker
        b = MT5Broker(cfg, login, password, server, terminal)
        if not b.connect():
            print("Connection FAILED. Check that MT5 is running and logged in.")
            return 1
        spec = b.symbol_spec(cfg.symbol.name)
        acct = b.account()
        tick = b.tick(cfg.symbol.name)
        print(f"\nAccount   : {acct.login} @ {acct.server}  demo={acct.is_demo}")
        print(f"Equity    : {acct.equity:,.2f} {acct.currency}")
        print(f"Symbol    : {spec.name}")
        print(f"  point={spec.point} digits={spec.digits} "
              f"contract={spec.contract_size}")
        print(f"  volume min/max/step = {spec.volume_min}/{spec.volume_max}"
              f"/{spec.volume_step}")
        print(f"  tick_value={spec.tick_value} tick_size={spec.tick_size}")
        print(f"  P/L per $1.00 move per 1.00 lot = "
              f"{spec.value_per_price_unit():.2f} {acct.currency}")
        print(f"  broker min stop distance = {spec.stops_level_points} points")
        print(f"Tick      : bid={tick.bid} ask={tick.ask} "
              f"spread={(tick.ask-tick.bid)/spec.point:.0f} points")
        b.disconnect()
        return 0

    from xauscalp.live import run_live
    run_live(cfg, login, password, server, terminal)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())