import sqlite3
import numpy as np
from pathlib import Path


class Database:
    def __init__(self, db_path: "str | Path"):
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        # Enable foreign-key enforcement so ON DELETE CASCADE works for
        # user_embeddings when a user row is deleted.
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL,
                embedding   BLOB NOT NULL,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            -- Individual per-capture embeddings for multi-embedding matching.
            -- Stored alongside the averaged embedding in users.embedding so that
            -- recognition can match against the closest single capture rather than
            -- a blended average, improving cross-device / cross-lighting recall.
            CREATE TABLE IF NOT EXISTS user_embeddings (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     INTEGER NOT NULL,
                embedding   BLOB NOT NULL,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
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

    def add_user(
        self,
        name: str,
        embedding: np.ndarray,
        individual_embeddings: "list[np.ndarray] | None" = None,
    ) -> int:
        cursor = self.conn.execute(
            "INSERT INTO users (name, embedding) VALUES (?, ?)",
            (name, embedding.tobytes()),
        )
        user_id = cursor.lastrowid
        if individual_embeddings:
            self.conn.executemany(
                "INSERT INTO user_embeddings (user_id, embedding) VALUES (?, ?)",
                [(user_id, e.tobytes()) for e in individual_embeddings],
            )
        self.conn.commit()
        return user_id

    def get_all_users(self) -> list:
        rows = self.conn.execute(
            "SELECT id, name, created_at FROM users ORDER BY id"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all_embeddings(self) -> list:
        """Return one entry per stored embedding.

        For users registered after multi-embedding support was added, each of
        their individual capture embeddings is returned as a separate entry
        (all sharing the same user_id/name).  For legacy users who only have
        the averaged embedding, that single embedding is returned instead.
        """
        users = self.conn.execute(
            "SELECT id, name, embedding FROM users ORDER BY id"
        ).fetchall()

        indiv_rows = self.conn.execute(
            "SELECT user_id, embedding FROM user_embeddings ORDER BY user_id, id"
        ).fetchall()

        # Build a map: user_id → [individual embeddings]
        indiv_map: dict[int, list] = {}
        for row in indiv_rows:
            indiv_map.setdefault(row["user_id"], []).append(
                np.frombuffer(row["embedding"], dtype=np.float32).copy()
            )

        result = []
        for u in users:
            uid, name = u["id"], u["name"]
            if uid in indiv_map:
                for emb in indiv_map[uid]:
                    result.append({"id": uid, "name": name, "embedding": emb})
            else:
                # Legacy user — fall back to averaged embedding
                result.append({
                    "id": uid,
                    "name": name,
                    "embedding": np.frombuffer(u["embedding"], dtype=np.float32).copy(),
                })
        return result

    def delete_user(self, user_id: int) -> bool:
        # Preserve historical access logs when a user is removed. Existing log
        # rows keep their matched_name/status/similarity, but their foreign key
        # is detached so the user row can be deleted safely.
        self.conn.execute(
            "UPDATE access_logs SET user_id = NULL WHERE user_id = ?",
            (user_id,),
        )
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

    def get_access_logs(self, limit: int = 20, offset: int = 0) -> list:
        rows = self.conn.execute(
            "SELECT id, user_id, matched_name, status, similarity, timestamp "
            "FROM access_logs ORDER BY timestamp DESC, id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [dict(r) for r in rows]

    def count_access_logs(self) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM access_logs"
        ).fetchone()[0]

    def close(self):
        self.conn.close()
