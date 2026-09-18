"""Decision journal.

Records every trade *and every rejected setup*. The rejection table is the more
valuable of the two: it tells you whether the system is passing on good trades
because a threshold is too tight, or correctly refusing to pay costs it cannot
recover. Without it you are tuning blind.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS session_plans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    session TEXT NOT NULL,
    session_open_utc TEXT NOT NULL,
    tradable INTEGER NOT NULL,
    regime_label TEXT,
    regime_score REAL,
    bias TEXT,
    allowed_setups TEXT,
    risk_scale REAL,
    payload TEXT
);

CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    session TEXT,
    setup TEXT,
    direction TEXT,
    entry REAL, stop REAL, tp1 REAL, tp2 REAL,
    quality REAL,
    r_tp1 REAL, r_tp2 REAL,
    accepted INTEGER,
    veto_reason TEXT,
    breakeven_wr REAL,
    cost_to_stop REAL,
    spread_points REAL,
    payload TEXT
);

CREATE TABLE IF NOT EXISTS rejections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    session TEXT,
    setup TEXT,
    reason TEXT,
    detail TEXT
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket INTEGER,
    opened_at TEXT,
    closed_at TEXT,
    session TEXT,
    setup TEXT,
    direction TEXT,
    volume REAL,
    entry_price REAL,
    exit_price REAL,
    stop REAL,
    pnl REAL,
    commission REAL,
    r_multiple REAL,
    exit_reason TEXT,
    payload TEXT
);

CREATE TABLE IF NOT EXISTS equity (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    balance REAL,
    equity REAL,
    open_positions INTEGER
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    level TEXT,
    kind TEXT,
    message TEXT,
    detail TEXT
);

CREATE INDEX IF NOT EXISTS idx_rej_ts ON rejections(ts);
CREATE INDEX IF NOT EXISTS idx_sig_ts ON signals(ts);
CREATE INDEX IF NOT EXISTS idx_trd_closed ON trades(closed_at);
"""


def _j(obj: Any) -> str:
    def default(o):
        if isinstance(o, datetime):
            return o.isoformat()
        return str(o)
    return json.dumps(obj, default=default)


class Journal:
    """Batched writer.

    Committing on every row means an fsync per row. Across a 120k-bar backtest
    that generates tens of thousands of rejection records, that is the difference
    between a run finishing and a run you give up on. Writes are batched and
    flushed every COMMIT_EVERY rows, on close, and on any read.
    """

    COMMIT_EVERY = 500

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # WAL plus relaxed sync: this is an analysis log, not a ledger. If the
        # process is killed mid-run the last few rows are expendable.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        with closing(self._conn.cursor()) as cur:
            cur.executescript(SCHEMA)
        self._conn.commit()
        self._pending = 0

    def _touch(self, n: int = 1) -> None:
        self._pending += n
        if self._pending >= self.COMMIT_EVERY:
            self._conn.commit()
            self._pending = 0

    def flush(self) -> None:
        if self._pending:
            self._conn.commit()
            self._pending = 0

    def close(self) -> None:
        self.flush()
        self._conn.close()

    # ------------------------------------------------------------------ writers
    def log_plan(self, plan) -> None:
        d = plan.to_dict()
        self._conn.execute(
            "INSERT INTO session_plans (created_at, session, session_open_utc, "
            "tradable, regime_label, regime_score, bias, allowed_setups, "
            "risk_scale, payload) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (d["created_at"], d["session"], d["session_open_utc"],
             int(d["tradable"]), d["regime"]["label"], d["regime"]["score"],
             d["bias"], ",".join(d["allowed_setups"]), d["risk_scale"], _j(d)),
        )
        self._touch()

    def log_signal(self, signal, accepted: bool, veto_reason: str = "",
                   breakeven_wr: float | None = None,
                   cost_to_stop: float | None = None,
                   spread_points: float | None = None) -> None:
        d = signal.to_dict()
        self._conn.execute(
            "INSERT INTO signals (ts, session, setup, direction, entry, stop, "
            "tp1, tp2, quality, r_tp1, r_tp2, accepted, veto_reason, "
            "breakeven_wr, cost_to_stop, spread_points, payload) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (d["ts"], d["session"], d["setup"], d["direction"], d["entry"],
             d["stop"], d["tp1"], d["tp2"], d["quality"], signal.r_to_tp1,
             signal.r_to_tp2, int(accepted), veto_reason, breakeven_wr,
             cost_to_stop, spread_points, _j(d)),
        )
        self._touch()

    def log_rejection(self, rej) -> None:
        self._conn.execute(
            "INSERT INTO rejections (ts, session, setup, reason, detail) "
            "VALUES (?,?,?,?,?)",
            (rej.ts.isoformat(), rej.session.value, rej.setup, rej.reason,
             _j(rej.detail)),
        )
        self._touch()

    def log_rejections(self, rejections) -> None:
        rows = [(r.ts.isoformat(), r.session.value, r.setup, r.reason, _j(r.detail))
                for r in rejections]
        if not rows:
            return
        self._conn.executemany(
            "INSERT INTO rejections (ts, session, setup, reason, detail) "
            "VALUES (?,?,?,?,?)", rows,
        )
        self._touch(len(rows))

    def log_trade(self, trade: dict) -> None:
        self._conn.execute(
            "INSERT INTO trades (ticket, opened_at, closed_at, session, setup, "
            "direction, volume, entry_price, exit_price, stop, pnl, commission, "
            "r_multiple, exit_reason, payload) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (trade.get("ticket"), trade.get("opened_at"), trade.get("closed_at"),
             trade.get("session"), trade.get("setup"), trade.get("direction"),
             trade.get("volume"), trade.get("entry_price"), trade.get("exit_price"),
             trade.get("stop"), trade.get("pnl"), trade.get("commission"),
             trade.get("r_multiple"), trade.get("exit_reason"), _j(trade)),
        )
        self._touch()

    def log_equity(self, ts: datetime, balance: float, equity: float,
                   open_positions: int) -> None:
        self._conn.execute(
            "INSERT INTO equity (ts, balance, equity, open_positions) VALUES (?,?,?,?)",
            (ts.isoformat(), balance, equity, open_positions),
        )
        self._touch()

    def log_event(self, ts: datetime, level: str, kind: str, message: str,
                  detail: dict | None = None) -> None:
        self._conn.execute(
            "INSERT INTO events (ts, level, kind, message, detail) VALUES (?,?,?,?,?)",
            (ts.isoformat(), level, kind, message, _j(detail or {})),
        )
        self._touch()

    # ------------------------------------------------------------------ readers
    def rejection_summary(self, limit: int = 30) -> list[tuple[str, str, int]]:
        self.flush()
        cur = self._conn.execute(
            "SELECT setup, reason, COUNT(*) c FROM rejections "
            "GROUP BY setup, reason ORDER BY c DESC LIMIT ?", (limit,),
        )
        return [(r["setup"], r["reason"], r["c"]) for r in cur.fetchall()]

    def trade_stats(self) -> dict:
        self.flush()
        cur = self._conn.execute(
            "SELECT COUNT(*) n, SUM(pnl) pnl, AVG(r_multiple) avg_r, "
            "SUM(CASE WHEN pnl>0 THEN 1 ELSE 0 END) wins FROM trades"
        )
        row = cur.fetchone()
        n = row["n"] or 0
        return {
            "trades": n,
            "net_pnl": row["pnl"] or 0.0,
            "avg_r": row["avg_r"] or 0.0,
            "win_rate": (row["wins"] / n) if n else 0.0,
        }