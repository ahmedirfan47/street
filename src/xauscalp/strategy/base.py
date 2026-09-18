"""Shared strategy types."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any

from ..analytics.regime import RegimeRead
from ..sessions import Session


@dataclass
class Signal:
    ts: datetime
    session: Session
    setup: str                 # "sweep" | "displacement" | "fib_*" | "orb" | "vwap"
    direction: str             # "long" | "short"
    entry: float
    stop: float
    tp1: float
    tp2: float
    reason: str
    quality: float             # 0..1 setup-intrinsic confidence
    level_name: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def stop_distance(self) -> float:
        return abs(self.entry - self.stop)

    @property
    def r_to_tp1(self) -> float:
        d = self.stop_distance
        return abs(self.tp1 - self.entry) / d if d > 0 else 0.0

    @property
    def r_to_tp2(self) -> float:
        d = self.stop_distance
        return abs(self.tp2 - self.entry) / d if d > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ts"] = self.ts.isoformat()
        d["session"] = self.session.value
        return d


@dataclass
class Rejection:
    """Why a candidate was not taken. Logged for every rejected setup -
    the rejection log is where you actually learn what the system is doing."""
    ts: datetime
    session: Session
    setup: str
    reason: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionPlan:
    """The pre-session analysis artefact, produced before each session opens."""
    created_at: datetime
    session: Session
    session_open_utc: datetime
    regime: RegimeRead
    bias: str                       # "up" | "down" | "neutral"
    allowed_setups: list[str]
    key_levels: list[dict[str, Any]]
    risk_scale: float
    max_trades: int
    direction: str = "both"      # "long" | "short" | "both" | "none"
    direction_conviction: float = 0.0
    direction_reason: str = ""
    notes: list[str] = field(default_factory=list)
    tradable: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at.isoformat(),
            "session": self.session.value,
            "session_open_utc": self.session_open_utc.isoformat(),
            "regime": self.regime.to_dict(),
            "bias": self.bias,
            "allowed_setups": self.allowed_setups,
            "key_levels": self.key_levels,
            "risk_scale": self.risk_scale,
            "max_trades": self.max_trades,
            "direction": self.direction,
            "direction_conviction": self.direction_conviction,
            "direction_reason": self.direction_reason,
            "notes": self.notes,
            "tradable": self.tradable,
        }

    def render(self) -> str:
        lines = [
            "",
            "=" * 74,
            f"  PRE-SESSION PLAN  |  {self.session.value.upper()}"
            f"  |  opens {self.session_open_utc:%Y-%m-%d %H:%M} UTC",
            "=" * 74,
            f"  Regime      : {self.regime.label}  (score {self.regime.score:.1f}/100)",
            f"  ATR         : {self.regime.atr:.2f}   pct-rank {self.regime.atr_pct:.2f}"
            f"   ADX {self.regime.adx:.1f}",
            f"  Structure   : {self.regime.structure}"
            f"   quality {self.regime.structure_quality:.2f}",
            f"  Spread      : health {self.regime.spread_health:.2f}",
            f"  Bias        : {self.bias}",
            f"  DIRECTION   : {self.direction.upper()}"
            f"   conviction {self.direction_conviction:.2f}"
            f"   [{self.direction_reason}]",
            f"  Setups      : {', '.join(self.allowed_setups) or 'NONE - stand down'}",
            f"  Risk scale  : {self.risk_scale:.2f}x    max trades {self.max_trades}",
            "  Key levels  :",
        ]
        for lv in self.key_levels[:10]:
            lines.append(f"      {lv['name']:<16} {lv['price']:>12.2f}  "
                         f"({lv['kind']}, w={lv['weight']:.2f})")
        if self.notes:
            lines.append("  Notes       :")
            lines.extend(f"      - {n}" for n in self.notes)
        lines.append("=" * 74)
        return "\n".join(lines)