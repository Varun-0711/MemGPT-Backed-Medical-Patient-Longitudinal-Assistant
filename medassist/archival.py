"""Archival storage with hybrid retrieval (Session 06A): BM25 + hashed char-trigram vectors + RRF.

Pure stdlib and deterministic across processes (crc32, not Python's randomized hash()).
"""
import json
import math
import re
import zlib
from collections import Counter

from . import db

STOP = {
    "the", "a", "an", "my", "me", "i", "was", "is", "what", "were", "of", "and", "to", "in", "on",
    "for", "last", "do", "did", "you", "tell", "about", "are", "am", "at", "it", "with", "have", "has",
    "can", "number",
}
DIM = 256


def tokenize(text: str) -> list:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP]


def embed(text: str) -> list:
    vec = [0.0] * DIM
    for w in tokenize(text):
        w = f"<{w}>"
        for i in range(len(w) - 2):
            vec[zlib.crc32(w[i:i + 3].encode()) % DIM] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine(a: list, b: list) -> float:
    return sum(x * y for x, y in zip(a, b))


def bm25_scores(query_tokens: list, docs: list, k1: float = 1.5, b: float = 0.75) -> list:
    n = len(docs)
    avgdl = (sum(len(d) for d in docs) / n) if n else 1.0
    df = Counter()
    for d in docs:
        for w in set(d):
            df[w] += 1
    scores = []
    for d in docs:
        counts, dl, s = Counter(d), len(d), 0.0
        for q in query_tokens:
            if q in counts:
                idf = math.log((n - df[q] + 0.5) / (df[q] + 0.5) + 1.0)
                tf = counts[q]
                s += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
        scores.append(s)
    return scores


def reciprocal_rank_fusion(rank_lists: list, k: int = 60) -> dict:
    fused = {}
    for ranked in rank_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return fused


class Archive:
    def __init__(self, conn, user_id: str):
        self.conn, self.user_id = conn, user_id

    def insert(self, text: str, kind: str = "note") -> int:
        return db.add_archival(self.conn, self.user_id, kind, text, json.dumps(embed(text)))

    def count(self) -> int:
        return len(db.all_archival(self.conn, self.user_id))

    def search(self, query: str, k: int = 3, include_conversation: bool = False) -> list:
        rows = db.all_archival(self.conn, self.user_id)
        if not include_conversation:
            rows = [row for row in rows if row["kind"] != "conversation"]
        if not rows:
            return []
        q_tokens = tokenize(query)
        sparse = bm25_scores(q_tokens, [tokenize(r["text"]) for r in rows])
        q_vec = embed(query)
        dense = [cosine(q_vec, json.loads(r["embedding"])) for r in rows]
        sparse_rank = [rows[i]["id"] for i in sorted(range(len(rows)), key=lambda i: -sparse[i]) if sparse[i] > 0]
        dense_rank = [rows[i]["id"] for i in sorted(range(len(rows)), key=lambda i: -dense[i])]
        fused = reciprocal_rank_fusion([sparse_rank, dense_rank])
        by_id = {r["id"]: (r, sparse[i], dense[i]) for i, r in enumerate(rows)}
        out = []
        for doc_id, score in sorted(fused.items(), key=lambda x: -x[1]):
            row, bm, cs = by_id[doc_id]
            if bm > 0 or cs >= 0.45:  # drop pure-noise dense matches
                out.append({"id": doc_id, "kind": row["kind"], "text": row["text"], "score": round(score, 5)})
        return out[:k]
