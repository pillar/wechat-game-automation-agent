"""In-process event bus for the live dashboard.

A global singleton; `emit(type, payload)` appends to a ring buffer and
broadcasts to all active subscriber queues. No-op when the bus is disabled
(default), so emit calls scattered in hot paths cost nothing in normal runs.
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DashboardBus:
    def __init__(self, ring_size: int = 500) -> None:
        self._enabled = False
        self._ring: deque = deque(maxlen=ring_size)
        self._subs: List[queue.Queue] = []
        self._lock = threading.Lock()
        self._seq = 0
        self._log_fp = None
        self._log_path: Optional[Path] = None

    def enable(self, log_path: Optional[Path] = None) -> None:
        self._enabled = True
        if log_path is not None and self._log_fp is None:
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                self._log_fp = open(log_path, "a", encoding="utf-8", buffering=1)
                self._log_path = log_path
                logger.info(f"[DASH] event log → {log_path}")
            except Exception as e:
                logger.warning(f"[DASH] failed to open event log {log_path}: {e}")

    def is_enabled(self) -> bool:
        return self._enabled

    def log_path(self) -> Optional[Path]:
        return self._log_path

    def emit(self, event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
        if not self._enabled:
            return
        with self._lock:
            self._seq += 1
            evt = {
                "seq": self._seq,
                "ts": time.time(),
                "type": event_type,
                "data": payload or {},
            }
            self._ring.append(evt)
            if self._log_fp is not None:
                try:
                    self._log_fp.write(json.dumps(evt, ensure_ascii=False) + "\n")
                except Exception:
                    pass
            dead: List[queue.Queue] = []
            for q in self._subs:
                try:
                    q.put_nowait(evt)
                except queue.Full:
                    dead.append(q)
            for q in dead:
                try:
                    self._subs.remove(q)
                except ValueError:
                    pass

    def subscribe(self, include_backlog: bool = True) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=1000)
        with self._lock:
            if include_backlog:
                for evt in list(self._ring):
                    try:
                        q.put_nowait(evt)
                    except queue.Full:
                        break
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            try:
                self._subs.remove(q)
            except ValueError:
                pass

    def snapshot(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._ring)


bus = DashboardBus()


def emit(event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
    bus.emit(event_type, payload)
