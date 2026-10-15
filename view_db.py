import sqlite3, sys

path = sys.argv[1] if len(sys.argv) > 1 else "medassist.db"
conn = sqlite3.connect(path)

tables = [r[0] for r in conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]

for t in tables:
    cols = [c[1] for c in conn.execute(f"PRAGMA table_info({t})")]
    if "embedding" in cols:  # skip the long vector column
        cols.remove("embedding")
    rows = conn.execute(f"SELECT {', '.join(cols)} FROM {t}").fetchall()
    print(f"\n=== {t} ({len(rows)} rows) ===")
    print(" | ".join(cols))
    for r in rows[:15]:
        print(" | ".join(str(x)[:90] for x in r))
    if len(rows) > 15:
        print(f"... {len(rows) - 15} more rows")