"""SQLite store for user feedback -> becomes the preference dataset for DPO."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from copilot.config import get_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    context TEXT NOT NULL,
    sources_json TEXT,
    provider TEXT,
    rating INTEGER,              -- +1 thumbs up, -1 thumbs down, NULL = no feedback
    corrected_answer TEXT        -- optional human-written better answer
)
"""


def _conn(db_path: str | None = None) -> sqlite3.Connection:
    path = Path(db_path or get_settings().feedback_db)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute(SCHEMA)
    return conn


def log_interaction(question, answer, context, sources=None, provider=None, db_path=None) -> int:
    with _conn(db_path) as c:
        cur = c.execute(
            "INSERT INTO interactions (ts, question, answer, context, sources_json, provider) VALUES (?,?,?,?,?,?)",
            (time.time(), question, answer, context, json.dumps(sources or []), provider),
        )
        return cur.lastrowid


def set_feedback(interaction_id: int, rating: int, corrected_answer: str | None = None, db_path=None) -> bool:
    if rating not in (-1, 1):
        raise ValueError("rating must be +1 or -1")
    with _conn(db_path) as c:
        cur = c.execute(
            "UPDATE interactions SET rating=?, corrected_answer=? WHERE id=?",
            (rating, (corrected_answer or "").strip() or None, interaction_id),
        )
        return cur.rowcount > 0


def build_preference_pairs(db_path=None) -> list[dict]:
    """A usable pair needs a rejected answer (thumbs-down) AND a human-corrected answer (chosen)."""
    from copilot.prompts import build_user_prompt

    with _conn(db_path) as c:
        rows = c.execute("SELECT * FROM interactions WHERE rating = -1 AND corrected_answer IS NOT NULL").fetchall()
    return [
        {
            "prompt": build_user_prompt(r["context"], r["question"]),
            "chosen": r["corrected_answer"],
            "rejected": r["answer"],
        }
        for r in rows
    ]
