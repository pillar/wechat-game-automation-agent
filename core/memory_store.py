"""SQLite-backed persistent memory: trajectories, skills, blacklist."""

import os
import sqlite3
import time
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_DB = Path("data") / "memory.db"


SCHEMA = """
CREATE TABLE IF NOT EXISTS trajectories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    ts REAL NOT NULL,
    scene TEXT,
    target TEXT,
    action_type TEXT,
    x INTEGER,
    y INTEGER,
    outcome TEXT,
    notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_traj_scene_target ON trajectories(scene, target);
CREATE INDEX IF NOT EXISTS idx_traj_session ON trajectories(session_id);

CREATE TABLE IF NOT EXISTS skills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scene TEXT NOT NULL,
    target_key TEXT NOT NULL,
    action_type TEXT NOT NULL,
    x INTEGER,
    y INTEGER,
    success_count INTEGER DEFAULT 0,
    fail_count INTEGER DEFAULT 0,
    last_updated REAL NOT NULL,
    UNIQUE(scene, target_key)
);
CREATE INDEX IF NOT EXISTS idx_skill_scene ON skills(scene);

CREATE TABLE IF NOT EXISTS blacklist (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scene TEXT NOT NULL,
    x INTEGER NOT NULL,
    y INTEGER NOT NULL,
    radius INTEGER NOT NULL,
    reason TEXT,
    created_at REAL NOT NULL,
    expires_at REAL
);
CREATE INDEX IF NOT EXISTS idx_bl_scene ON blacklist(scene);
"""


class MemoryStore:
    """Persistent memory across sessions.

    Records every action with outcome (for learning), caches successful
    scene+target → coord skills, and holds a persistent blacklist separate
    from the in-memory short-term one in the game adapter.
    """

    def __init__(self, db_path: Optional[str] = None, session_id: Optional[str] = None,
                 skill_min_success: int = 1):
        self.db_path = Path(db_path or os.environ.get("MEMORY_DB", _DEFAULT_DB))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or f"sess-{int(time.time())}"
        self.skill_min_success = max(1, int(skill_min_success))
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        logger.info(f"[MEM] session={self.session_id} db={self.db_path} skill_min_success={self.skill_min_success}")

    def record_action(
        self,
        scene: str,
        target: str,
        action_type: str,
        x: Optional[int],
        y: Optional[int],
        outcome: str,
        notes: str = "",
    ) -> None:
        self._conn.execute(
            "INSERT INTO trajectories(session_id, ts, scene, target, action_type, x, y, outcome, notes) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (self.session_id, time.time(), scene, target, action_type, x, y, outcome, notes),
        )
        self._conn.commit()

        if outcome in ("success", "changed"):
            self._upsert_skill(scene, target, action_type, x, y, success=True)
        elif outcome in ("no_change", "rejected", "stuck"):
            self._upsert_skill(scene, target, action_type, x, y, success=False)

    def _upsert_skill(self, scene: str, target: str, action_type: str,
                      x: Optional[int], y: Optional[int], success: bool) -> None:
        key = self._normalize_target(target)
        row = self._conn.execute(
            "SELECT id, success_count, fail_count FROM skills WHERE scene=? AND target_key=?",
            (scene, key),
        ).fetchone()
        now = time.time()
        if row is None:
            self._conn.execute(
                "INSERT INTO skills(scene, target_key, action_type, x, y, success_count, fail_count, last_updated) "
                "VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
                (scene, key, action_type, x, y, 1 if success else 0, 0 if success else 1, now),
            )
        else:
            sid, sc, fc = row
            new_sc = sc + (1 if success else 0)
            new_fc = fc + (0 if success else 1)
            update_coord = success and x is not None and y is not None
            if update_coord:
                self._conn.execute(
                    "UPDATE skills SET x=?, y=?, success_count=?, fail_count=?, last_updated=? WHERE id=?",
                    (x, y, new_sc, new_fc, now, sid),
                )
            else:
                self._conn.execute(
                    "UPDATE skills SET success_count=?, fail_count=?, last_updated=? WHERE id=?",
                    (new_sc, new_fc, now, sid),
                )
        self._conn.commit()

    def lookup_skill(self, scene: str, target: str) -> Optional[Dict[str, Any]]:
        key = self._normalize_target(target)
        row = self._conn.execute(
            "SELECT action_type, x, y, success_count, fail_count FROM skills "
            "WHERE scene=? AND target_key=? AND success_count >= ? AND success_count > fail_count",
            (scene, key, self.skill_min_success),
        ).fetchone()
        if row is None:
            return None
        action_type, x, y, sc, fc = row
        return {
            "action_type": action_type,
            "x": x,
            "y": y,
            "success_count": sc,
            "fail_count": fc,
        }

    def add_blacklist(self, scene: str, x: int, y: int, radius: int = 60,
                      reason: str = "", ttl_s: Optional[float] = None) -> None:
        now = time.time()
        expires = (now + ttl_s) if ttl_s else None
        self._conn.execute(
            "INSERT INTO blacklist(scene, x, y, radius, reason, created_at, expires_at) "
            "VALUES(?, ?, ?, ?, ?, ?, ?)",
            (scene, x, y, radius, reason, now, expires),
        )
        self._conn.commit()

    def is_blacklisted(self, scene: str, x: int, y: int) -> Optional[str]:
        now = time.time()
        rows = self._conn.execute(
            "SELECT x, y, radius, reason, expires_at FROM blacklist WHERE scene=?",
            (scene,),
        ).fetchall()
        for bx, by, br, reason, exp in rows:
            if exp is not None and exp < now:
                continue
            if abs(x - bx) <= br and abs(y - by) <= br:
                return reason or "blacklisted"
        return None

    def purge_expired_blacklist(self) -> int:
        now = time.time()
        cur = self._conn.execute(
            "DELETE FROM blacklist WHERE expires_at IS NOT NULL AND expires_at < ?",
            (now,),
        )
        self._conn.commit()
        return cur.rowcount

    def recent_failures(self, scene: str, target: str, limit: int = 5) -> List[Tuple[int, int]]:
        key = self._normalize_target(target)
        rows = self._conn.execute(
            "SELECT x, y FROM trajectories WHERE scene=? AND target=? AND outcome IN ('no_change','rejected','stuck') "
            "ORDER BY ts DESC LIMIT ?",
            (scene, key, limit),
        ).fetchall()
        return [(r[0], r[1]) for r in rows if r[0] is not None]

    @staticmethod
    def _normalize_target(target: str) -> str:
        if not target:
            return ""
        return target.strip()[:80]

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
