"""Sanity checks. Run: python check.py

Queries MT5 for the REAL contract spec when the terminal is running. Falls back
to a generic 100oz spec only when it cannot connect, and says so loudly - an
earlier version silently used the fallback and reported 100 USD per price unit
on a broker whose actual figure was 10.
"""
import sys
sys.path.insert(0, "src")

from xauscalp.config import load_config
from xauscalp.risk.sizing import (
    SymbolSpec, breakeven_win_rate, estimate_costs, position_size,
)
from xauscalp.strategy.fibonacci import ImpulseLeg

cfg = load_config("config/default.yaml")

# ---- real spec from MT5 if we can get it --------------------------------
spec = None
live_spread = None
source = "FALLBACK (generic 100oz contract)"
try:
    from xauscalp.broker.mt5_broker import MT5Broker
    b = MT5Broker(cfg)
    if b.connect():
        spec = b.symbol_spec(cfg.symbol.name)
        t = b.tick(cfg.symbol.name)
        live_spread = (t.ask - t.bid) / spec.point
        source = f"LIVE from MT5 ({b.account().server})"
        b.disconnect()
except Exception as exc:
    print(f"  (MT5 unavailable: {exc})")

if spec is None:
    s = cfg.symbol
    spec = SymbolSpec(
        name=s.name, point=s.fallback_point, digits=s.fallback_digits,
        contract_size=s.fallback_contract_size, volume_min=s.fallback_volume_min,
        volume_max=s.fallback_volume_max, volume_step=s.fallback_volume_step,
        tick_value=s.fallback_contract_size * s.fallback_point,
        tick_size=s.fallback_point,
    )

vpu = spec.value_per_price_unit()
spread = live_spread if live_spread is not None else cfg.backtest.spread_points_mean

print("=" * 62)
print(f"CONTRACT   [{source}]")
print("=" * 62)
print(f"  contract_size = {spec.contract_size}   tick_value = {spec.tick_value}")
print(f"  P/L per 1.00 price unit per 1.00 lot : {vpu:.2f} USD")
if abs(vpu - 100.0) < 1e-6:
    print("  -> standard 100oz contract")
elif abs(vpu - 10.0) < 1e-6:
    print("  -> 10oz-equivalent contract. Commission must be scaled to match:")
    print("     ~0.35/lot/side here, NOT the ~3.50 a 100oz contract implies.")
else:
    print(f"  -> UNUSUAL. Verify with your broker before trading.")

print()
print("=" * 62)
print(f"COSTS  (spread {spread:.0f} points)")
print("=" * 62)
e = estimate_costs(cfg, spec, spread)
print(f"  spread      : {e.spread_cost:.2f}")
print(f"  commission  : {e.commission_price_units:.2f}")
print(f"  slippage    : {e.slippage_cost:.2f}")
print(f"  ROUND TURN  : {e.total_round_turn:.3f}")
min_stop = e.total_round_turn / cfg.costs.max_cost_to_stop_ratio
print(f"  min viable stop at {cfg.costs.max_cost_to_stop_ratio:.0%} ratio : {min_stop:.2f}")
print("  -> every setup widens its stop to at least this")

print()
print("=" * 62)
print(f"SIZING  ({cfg.backtest.initial_equity:,.0f} equity, "
      f"{cfg.risk.risk_pct_per_trade}% per trade)")
print("=" * 62)
for stop in (min_stop, 8.0, 15.0):
    v, r = position_size(cfg, spec, cfg.backtest.initial_equity, stop)
    print(f"  stop {stop:6.2f} -> {v:6.2f} lots, risk {r:7.2f}")
print(f"  portfolio cap {cfg.risk.max_portfolio_risk_pct}% = "
      f"{cfg.backtest.initial_equity * cfg.risk.max_portfolio_risk_pct / 100:.2f} total open risk")
print(f"  breakeven win rate (1R/2.4R, 50% partial): "
      f"{breakeven_win_rate(1.0, 2.4, 50.0, max(min_stop, 8.0), e) * 100:.1f}%")

print()
print("=" * 62)
print("FIBONACCI GEOMETRY  (40.00 up leg from 4400)")
print("=" * 62)
leg = ImpulseLeg(0, 10, 4400.0, 4440.0, "long")
for f in (0.382, 0.50, 0.618, 0.705, 0.786):
    print(f"  retrace {f:.3f}   : {leg.retracement(f):.2f}")
for f in (1.0, 1.272, 1.618):
    print(f"  extension {f:.3f} : {leg.extension(f):.2f}")
print()
print("  Profile stops on this leg (ATR assumed 7.0):")
for p in cfg.fib.profiles:
    if not p.enabled:
        print(f"    {p.name:<12} DISABLED")
        continue
    entry = leg.retracement(p.entry_zone_start)
    stop = leg.retracement(p.stop_level) - p.sl_buffer_atr * 7.0
    d = abs(entry - stop)
    ratio = e.total_round_turn / d if d > 0 else 999
    ok = "OK" if ratio <= cfg.costs.max_cost_to_stop_ratio else "REJECTED by cost gate"
    print(f"    {p.name:<12} entry {entry:8.2f}  stop {stop:8.2f}  "
          f"dist {d:5.2f}  cost {ratio:5.1%}  {ok}")

print()
print("=" * 62)
print("CONFIG")
print("=" * 62)
print(f"  symbol     : {cfg.symbol.name}")
print(f"  timeframes : {cfg.execution.bar_timeframe} signal / "
      f"{cfg.execution.htf_timeframe} context / {cfg.execution.bias_timeframe} bias")
print(f"  mode       : {cfg.execution.mode}   live-on-real: "
      f"{cfg.execution.allow_live_on_real_account}")
print(f"  server offset override : {cfg.execution.server_utc_offset_hours}")
print(f"  directional lock : {cfg.sessions.directional_lock}   "
      f"min conviction {cfg.sessions.direction_min_conviction}")
print(f"  concurrency : {cfg.risk.max_concurrent_positions}   "
      f"session cap {cfg.risk.max_trades_per_session}   "
      f"day cap {cfg.risk.max_trades_per_day}")
enabled = [n for n, on in [
    ("sweep", cfg.sweep.enabled), ("displacement", cfg.displacement.enabled),
    ("orb", cfg.orb.enabled), ("vwap", cfg.vwap.enabled),
] if on] + [f"fib_{p.name}" for p in cfg.fib.profiles if p.enabled]
print(f"  setups live : {', '.join(enabled)}")
print()