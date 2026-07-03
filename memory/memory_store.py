"""
Long-term memory for personalization.

Stores two things per user:
1. `preferences` - durable facts about the user (key/value), e.g. "favorite_topic": "AI"
2. `history`     - rolling window of recent (role, content) turns for context

Implemented with SQLite for zero-setup local development. The interface
(`MemoryStore`) is intentionally storage-agnostic - swap the body of each
method for pymongo calls to back it with MongoDB instead, without touching
any calling code.
"""
import sqlite3
import json
import os
import time
from config import settings


class MemoryStore:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or settings.MEMORY_DB_PATH
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._connect() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS preferences (
                    user_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (user_id, key)
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at REAL NOT NULL
                )"""
            )

    # ---------- preferences ----------
    def set_preference(self, user_id: str, key: str, value: str):
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO preferences (user_id, key, value, updated_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(user_id, key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at""",
                (user_id, key, value, time.time()),
            )

    def get_preferences(self, user_id: str) -> dict:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT key, value FROM preferences WHERE user_id = ?", (user_id,)
            ).fetchall()
        return {k: v for k, v in rows}

    # ---------- history ----------
    def add_turn(self, user_id: str, role: str, content: str):
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO history (user_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (user_id, role, content, time.time()),
            )

    def get_recent_history(self, user_id: str, limit: int = None) -> list[dict]:
        limit = limit or settings.MEMORY_HISTORY_WINDOW
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT role, content FROM history WHERE user_id = ?
                   ORDER BY id DESC LIMIT ?""",
                (user_id, limit),
            ).fetchall()
        return [{"role": r, "content": c} for r, c in reversed(rows)]

    # ---------- convenience ----------
    def get_user_context(self, user_id: str) -> dict:
        """Bundle preferences + recent history for the LangGraph memory node."""
        return {
            "preferences": self.get_preferences(user_id),
            "history": self.get_recent_history(user_id),
        }


def extract_and_store_preferences(memory: MemoryStore, user_id: str, message: str):
    """
    Lightweight rule-based preference capture (e.g. "I like X", "my name is Y").
    For production, replace with an LLM call that returns structured key/value pairs.
    """
    lowered = message.lower()
    if "my name is" in lowered:
        name = message.split("my name is", 1)[-1].strip().split(".")[0].split(",")[0]
        if name:
            memory.set_preference(user_id, "name", name.strip())
    if "i like" in lowered:
        liked = message.lower().split("i like", 1)[-1].strip().split(".")[0]
        if liked:
            memory.set_preference(user_id, "likes", liked.strip())
