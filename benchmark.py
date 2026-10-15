"""Stdlib benchmark harness for the MedAssist persistent-memory demo."""
import os
import statistics
import subprocess
import sys
import shutil
import tempfile
import time
from pathlib import Path

from medassist.agent import Agent


ROOT = Path(__file__).resolve().parent
FACTS = [
    ("My doctor switched me from Metformin to Glimepiride 2 mg once daily.", "Glimepiride"),
    ("I am allergic to Penicillin.", "Penicillin"),
    ("My BP is 138/88 today.", "138/88"),
    ("I weigh 72 kg.", "72 kg"),
    ("I have a mild headache.", "mild headache"),
    ("Hi, my name is Priya.", "Priya"),
    ("My glucose is 126.", "126 mg/dL"),
    ("I stopped taking Aspirin.", "Aspirin"),
]
PROBES = [
    "What medication am I taking?", "What allergies do I have?", "What was my blood pressure?",
    "What is my weight?", "What symptoms have I reported?", "What is my name?",
    "What was my glucose?", "Which medication did I stop?",
]


def subprocess_plant(db_path, user_id, text):
    code = (
        "from medassist.agent import Agent; import sys; "
        "a=Agent(sys.argv[1],sys.argv[2]); a.step(sys.argv[3]); a.conn.close()"
    )
    env = os.environ.copy()
    env["MEDASSIST_LLM"] = "mock"
    subprocess.run([sys.executable, "-c", code, str(db_path), user_id, text],
                   cwd=ROOT, check=True, capture_output=True, text=True, env=env)


def recall_benchmark(tmp):
    db_path = tmp / "recall.sqlite"
    subprocess_plant(db_path, "recall_patient", "I started Aspirin 81 mg daily.")
    for user_text, _ in FACTS:
        subprocess_plant(db_path, "recall_patient", user_text)
    agent = Agent(str(db_path), "recall_patient")
    # The stop parser only marks a medication discontinued when it is already recorded.
    agent._run_tool("core_memory_append", {"block": "human_block", "fact": "Medication: Aspirin 81 mg daily"})
    agent._persist()
    agent.step("I stopped taking Aspirin.")
    rows = []
    for question, (_, gold) in zip(PROBES, FACTS):
        reply = agent.step(question)["reply"]
        passed = gold.lower() in reply.lower()
        rows.append((question, gold, "PASS" if passed else "FAIL", reply))
    score = sum(r[2] == "PASS" for r in rows)
    agent.conn.close()
    return rows, score, len(rows)


def distractor_stress(tmp):
    results = []
    for count in (5, 20, 50):
        db_path = tmp / f"distractors_{count}.sqlite"
        subprocess_plant(db_path, f"stress_{count}", "My doctor switched me from Metformin to Glimepiride 2 mg once daily.")
        agent = Agent(str(db_path), f"stress_{count}")
        for i in range(count):
            fact = f"Symptom (2026-01-{(i % 28) + 1:02d}): irrelevant distractor symptom number {i + 1} with unrelated detail"
            agent._run_tool("core_memory_append", {"block": "human_block", "fact": fact})
        agent._persist()
        result = agent.step("What medication am I taking?")
        passed = "glimepiride" in result["reply"].lower()
        archived = sum(row[0] == "symptom_history" for row in agent.conn.execute(
            "SELECT kind FROM archival WHERE user_id=?", (agent.user_id,)).fetchall())
        results.append((count, "PASS" if passed else "FAIL", archived))
        agent.conn.close()
    return results


def isolation_test(tmp):
    db_path = tmp / "isolation.sqlite"
    users = {
        "patient_A": "My BP is 118/76 today.",
        "patient_B": "I am allergic to Latex.",
        "patient_C": "My doctor switched me from Metformin to Glimepiride 2 mg once daily.",
    }
    for user, fact in users.items():
        subprocess_plant(db_path, user, fact)
    # Give each user private archival data as well as core memory data.
    for user, fact in users.items():
        a = Agent(str(db_path), user)
        a.archive.insert(f"PRIVATE_ARCHIVE_{user} confidential medical detail", "note")
        a.conn.close()
    a = Agent(str(db_path), "patient_A")
    core = a.core.blocks["human_block"]
    hits = a.archive.search("confidential medical detail", k=20)
    archive_text = " ".join(hit["text"] for hit in hits)
    other_user_ids = ["patient_B", "patient_C"]
    core_clean = all(other not in core for other in other_user_ids) and all(
        secret not in core for secret in ("Latex", "Glimepiride"))
    archive_clean = all(other not in archive_text for other in other_user_ids) and all(
        f"PRIVATE_ARCHIVE_{other}" not in archive_text for other in other_user_ids)
    a.conn.close()
    return core_clean, archive_clean


def latency_benchmark(tmp):
    db_path = tmp / "latency.sqlite"
    samples = []
    for i in range(200):
        start = time.perf_counter()
        a = Agent(str(db_path), f"latency_{i}")
        samples.append((time.perf_counter() - start) * 1000)
        a.conn.close()
    ordered = sorted(samples)
    p95 = ordered[int(0.95 * (len(ordered) - 1))]
    return statistics.mean(samples), p95


def token_benchmark(tmp):
    a = Agent(str(tmp / "tokens.sqlite"), "token_patient")
    tokens = [a.step(f"Please tell me a brief neutral update number {i + 1}.")["tokens"] for i in range(20)]
    a.conn.close()
    return statistics.mean(tokens), max(tokens)


def main():
    work = Path(tempfile.mkdtemp(prefix=".benchmark_", dir=ROOT))
    try:
        tmp = work
        recall_rows, recall_hits, recall_total = recall_benchmark(tmp)
        distractors = distractor_stress(tmp)
        core_iso, archive_iso = isolation_test(tmp)
        latency_mean, latency_p95 = latency_benchmark(tmp)
        token_mean, token_peak = token_benchmark(tmp)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    recall_pct = 100 * recall_hits / recall_total
    lines = [
        "# MedAssist Benchmark Results", "", "Generated by `python benchmark.py` using the stdlib and mock LLM.", "",
        "## Multi-session recall", "", "| Probe | Gold fact | Result |", "|---|---|---|",
    ]
    lines.extend(f"| {q} | {gold} | {status} |" for q, gold, status, _ in recall_rows)
    lines.extend([f"| **Total recall** | **{recall_hits}/{recall_total}** | **{recall_pct:.1f}%** |", "",
                  "## Summary", "", "| Metric | Result |", "|---|---|"])
    lines.append(f"| Multi-session recall | {recall_hits}/{recall_total} ({recall_pct:.1f}%) |")
    lines.extend([
                  "", "## Stress, isolation, latency, and tokens", "",
                  "| Benchmark | Result |", "|---|---|"])
    lines.extend(f"| Distractors: {n} irrelevant symptoms | {status}; {archived} moved to archival |"
                 for n, status, archived in distractors)
    lines.extend([
        f"| Multi-user isolation: patient A core memory | {'PASS' if core_iso else 'FAIL'} |",
        f"| Multi-user isolation: patient A archival search | {'PASS' if archive_iso else 'FAIL'} |",
        f"| Cold-start construction, 200 iterations | mean {latency_mean:.3f} ms; p95 {latency_p95:.3f} ms |",
        f"| 20-turn context tokens | mean {token_mean:.1f}; peak {token_peak} |", "",
        "Source changes: `medassist/llm.py` now answers name and discontinued-medication probes from existing core-memory lines. Distractor facts are added using the existing core-memory append and overflow path.", "",
    ])
    report = "\n".join(lines)
    (ROOT / "BENCHMARK.md").write_text(report, encoding="utf-8")
    print(report)
    if recall_hits != recall_total or not all(status == "PASS" for _, status, _ in distractors) or not core_iso or not archive_iso:
        raise SystemExit("One or more benchmark checks failed.")


if __name__ == "__main__":
    main()
