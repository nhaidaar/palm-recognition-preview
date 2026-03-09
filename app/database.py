import sqlite3
import numpy as np
from pathlib import Path


class Database:
    def __init__(self, db_path: "str | Path"):
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL,
                embedding   BLOB NOT NULL,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS access_logs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id         INTEGER,
                matched_name    TEXT NOT NULL,
                status          TEXT NOT NULL,
                similarity      REAL NOT NULL,
                timestamp       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
        """)
        self.conn.commit()

    def add_user(self, name: str, embedding: np.ndarray) -> int:
        cursor = self.conn.execute(
            "INSERT INTO users (name, embedding) VALUES (?, ?)",
            (name, embedding.tobytes()),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_all_users(self) -> list:
        rows = self.conn.execute(
            "SELECT id, name, created_at FROM users ORDER BY id"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all_embeddings(self) -> list:
        rows = self.conn.execute(
            "SELECT id, name, embedding FROM users ORDER BY id"
        ).fetchall()
        return [
            {
                "id": r["id"],
                "name": r["name"],
                "embedding": np.frombuffer(r["embedding"], dtype=np.float32).copy(),
            }
            for r in rows
        ]

    def delete_user(self, user_id: int) -> bool:
        cursor = self.conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    def add_access_log(
        self,
        user_id,
        matched_name: str,
        status: str,
        similarity: float,
    ):
        self.conn.execute(
            "INSERT INTO access_logs (user_id, matched_name, status, similarity) VALUES (?, ?, ?, ?)",
            (user_id, matched_name, status, similarity),
        )
        self.conn.commit()

    def get_access_logs(self, limit: int = 50) -> list:
        rows = self.conn.execute(
            "SELECT id, user_id, matched_name, status, similarity, timestamp "
            "FROM access_logs ORDER BY timestamp DESC, id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self):
        self.conn.close()
