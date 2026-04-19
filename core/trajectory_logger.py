"""JSONL trajectory logger for offline learning (DPO / SFT export)."""

import os
import json
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class TrajectoryLogger:
    """Append one JSON record per inference+action+outcome.

    Schema (one JSON object per line):
        ts, session_id, scene, model_family, prompt, response,
        parsed_action, executed, outcome, verify_reason

    Files: data/trajectories/<session_id>.jsonl
    """

    def __init__(self, session_id: Optional[str] = None, out_dir: Optional[str] = None):
        self.session_id = session_id or f"sess-{int(time.time())}"
        self.out_dir = Path(out_dir or os.environ.get("TRAJ_DIR", Path("data") / "trajectories"))
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.out_dir / f"{self.session_id}.jsonl"
        self._fp = open(self.path, "a", encoding="utf-8")
        logger.info(f"[TRAJ] writing to {self.path}")

    def log(
        self,
        scene: Optional[str],
        model_family: str,
        prompt: str,
        response: str,
        parsed_action: Optional[Dict[str, Any]],
        executed: bool,
        outcome: str,
        verify_reason: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        rec = {
            "ts": time.time(),
            "session_id": self.session_id,
            "scene": scene,
            "model_family": model_family,
            "prompt": prompt,
            "response": response,
            "parsed_action": parsed_action,
            "executed": executed,
            "outcome": outcome,
            "verify_reason": verify_reason,
        }
        if extra:
            rec["extra"] = extra
        try:
            self._fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._fp.flush()
        except Exception as e:
            logger.debug(f"[TRAJ] write failed: {e}")

    def close(self) -> None:
        try:
            self._fp.close()
        except Exception:
            pass
