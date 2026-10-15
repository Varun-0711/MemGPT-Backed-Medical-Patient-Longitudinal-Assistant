"""SQLite relational persistence (Session 08A schema, extended with archival + paged flag)."""
import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    user_id TEXT,
    role TEXT,
    content TEXT,
    paged INTEGER DEFAULT 0,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS core_memory (
    user_id TEXT,
    block_name TEXT,
    content TEXT,
    PRIMARY KEY (user_id, block_name)
);
CREATE TABLE IF NOT EXISTS archival (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    kind TEXT,
    text TEXT,
    embedding TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS escalations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    risk TEXT NOT NULL,
    reason TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_msg_session ON messages(session_id, paged);
CREATE INDEX IF NOT EXISTS idx_arch_user ON archival(user_id);
CREATE INDEX IF NOT EXISTS idx_escalation_user ON escalations(user_id, timestamp);
"""


def connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def create_session(conn, session_id: str, user_id: str):
    with conn:
        conn.execute("INSERT INTO sessions (session_id, user_id) VALUES (?, ?)", (session_id, user_id))


def set_core_memory(conn, user_id: str, block: str, content: str):
    with conn:
        conn.execute(
            """INSERT INTO core_memory (user_id, block_name, content) VALUES (?, ?, ?)
               ON CONFLICT(user_id, block_name) DO UPDATE SET content=excluded.content""",
            (user_id, block, content),
        )


def get_core_memory(conn, user_id: str) -> dict:
    cur = conn.execute("SELECT block_name, content FROM core_memory WHERE user_id=?", (user_id,))
    return {r[0]: r[1] for r in cur.fetchall()}


def log_turn(conn, session_id: str, user_id: str, role: str, content: str) -> int:
    with conn:
        cur = conn.execute(
            "INSERT INTO messages (session_id, user_id, role, content) VALUES (?, ?, ?, ?)",
            (session_id, user_id, role, content),
        )
    return cur.lastrowid


def active_messages(conn, session_id: str) -> list:
    cur = conn.execute(
        "SELECT id, role, content FROM messages WHERE session_id=? AND paged=0 ORDER BY id", (session_id,)
    )
    return cur.fetchall()


def mark_paged(conn, ids: list):
    if not ids:
        return
    with conn:
        conn.executemany("UPDATE messages SET paged=1 WHERE id=?", [(i,) for i in ids])


def add_archival(conn, user_id: str, kind: str, text: str, embedding_json: str) -> int:
    with conn:
        cur = conn.execute(
            "INSERT INTO archival (user_id, kind, text, embedding) VALUES (?, ?, ?, ?)",
            (user_id, kind, text, embedding_json),
        )
    return cur.lastrowid


def all_archival(conn, user_id: str) -> list:
    cur = conn.execute("SELECT id, kind, text, embedding FROM archival WHERE user_id=?", (user_id,))
    return [{"id": r[0], "kind": r[1], "text": r[2], "embedding": r[3]} for r in cur.fetchall()]


def log_escalation(conn, user_id: str, session_id: str, risk: str, reason: str):
    with conn:
        conn.execute(
            "INSERT INTO escalations (user_id, session_id, risk, reason) VALUES (?, ?, ?, ?)",
            (user_id, session_id, risk, reason),
        )
