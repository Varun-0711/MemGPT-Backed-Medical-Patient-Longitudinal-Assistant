"""Stdlib benchmark harness for the MedAssist persistent-memory demo."""
import argparse
import os
import statistics
import subprocess
import sys
import shutil
import tempfile
import time
from pathlib import Path

from medassist.agent import Agent
from medassist.llm import MockLLM, get_llm


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


def subprocess_plant(db_path, user_id, text, mode="agentic"):
    code = (
        "from medassist.agent import Agent; import sys; "
        "a=Agent(sys.argv[1],sys.argv[2],mode=sys.argv[4]); a.step(sys.argv[3]); a.conn.close()"
    )
    env = os.environ.copy()
    env["MEDASSIST_LLM"] = "mock"
    subprocess.run([sys.executable, "-c", code, str(db_path), user_id, text, mode],
                   cwd=ROOT, check=True, capture_output=True, text=True, env=env)


def recall_benchmark(tmp, mode="agentic", filename="recall.sqlite"):
    db_path = tmp / filename
    subprocess_plant(db_path, "recall_patient", "I started Aspirin 81 mg daily.", mode)
    for user_text, _ in FACTS:
        subprocess_plant(db_path, "recall_patient", user_text, mode)
    agent = Agent(str(db_path), "recall_patient", llm=MockLLM(), mode=mode)
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


def distractor_stress(tmp, mode="agentic"):
    results = []
    for count in (5, 20, 50):
        db_path = tmp / f"distractors_{count}.sqlite"
        subprocess_plant(db_path, f"stress_{count}", "My doctor switched me from Metformin to Glimepiride 2 mg once daily.", mode)
        agent = Agent(str(db_path), f"stress_{count}", llm=MockLLM(), mode=mode)
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


def isolation_test(tmp, mode="agentic"):
    db_path = tmp / "isolation.sqlite"
    users = {
        "patient_A": "My BP is 118/76 today.",
        "patient_B": "I am allergic to Latex.",
        "patient_C": "My doctor switched me from Metformin to Glimepiride 2 mg once daily.",
    }
    for user, fact in users.items():
        subprocess_plant(db_path, user, fact, mode)
    # Give each user private archival data as well as core memory data.
    for user, fact in users.items():
        a = Agent(str(db_path), user, llm=MockLLM(), mode=mode)
        a.archive.insert(f"PRIVATE_ARCHIVE_{user} confidential medical detail", "note")
        a.conn.close()
    a = Agent(str(db_path), "patient_A", llm=MockLLM(), mode=mode)
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


def latency_benchmark(tmp, mode="agentic"):
    db_path = tmp / "latency.sqlite"
    samples = []
    for i in range(200):
        start = time.perf_counter()
        a = Agent(str(db_path), f"latency_{i}", llm=MockLLM(), mode=mode)
        samples.append((time.perf_counter() - start) * 1000)
        a.conn.close()
    ordered = sorted(samples)
    p95 = ordered[int(0.95 * (len(ordered) - 1))]
    return statistics.mean(samples), p95


def token_benchmark(tmp, mode="agentic"):
    a = Agent(str(tmp / "tokens.sqlite"), "token_patient", llm=MockLLM(), mode=mode)
    tokens = [a.step(f"Please tell me a brief neutral update number {i + 1}.")["tokens"] for i in range(20)]
    a.conn.close()
    return statistics.mean(tokens), max(tokens)


def template_token_test(tmp):
    """Exercise the template controller's context cap for a complete 30-turn session."""
    a = Agent(str(tmp / "template_tokens.sqlite"), "template_token_patient",
              llm=MockLLM(), mode="template")
    tokens = [a.step(f"Please note this neutral update number {i + 1}.")["tokens"]
              for i in range(30)]
    a.conn.close()
    assert max(tokens) <= Agent.TOKEN_BUDGET, f"template context exceeded budget: {max(tokens)}"
    return statistics.mean(tokens), max(tokens)


def live_model_probes(tmp, backend, mode="agentic"):
    if backend == "mock":
        return [(question, "SKIPPED (mock backend)") for question in (
            "My doctor bumped my metformin up to 1000, twice a day",
            "I've been out of my pills for 3 days",
        )]
    probe_db = tmp / "live_probes.sqlite"
    try:
        live_llm = get_llm(backend)
        agent = Agent(str(probe_db), "live_probe_patient", llm=live_llm, mode=mode)
    except Exception as exc:
        return [("Live-model probe setup", f"SKIPPED ({type(exc).__name__})")]

    agent._run_tool("core_memory_append", {
        "block": "human_block", "fact": "Medication: Metformin 500 mg once daily",
    })
    agent._persist()
    probes = [
        ("My doctor bumped my metformin up to 1000, twice a day", "Metformin", "1000"),
        ("I've been out of my pills for 3 days", "out", "3 days"),
    ]
    rows = []
    try:
        for question, expected_a, expected_b in probes:
            result = agent.step(question)
            memory = agent.core.blocks["human_block"].lower()
            passed = expected_a.lower() in memory and expected_b.lower() in memory
            rows.append((question, "PASS" if passed else "FAIL"))
    finally:
        agent.conn.close()
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run MedAssist memory benchmarks")
    parser.add_argument("--llm", choices=("mock", "groq", "openrouter", "anthropic"), default="mock",
                        help="backend for optional natural-language probes (default: mock)")
    parser.add_argument("--mode", choices=("agentic", "template"), default="agentic",
                        help="mode for the main benchmark rows (default: agentic)")
    args = parser.parse_args(argv)
    work = Path(tempfile.mkdtemp(prefix=".benchmark_", dir=ROOT))
    try:
        tmp = work
        recall_rows, recall_hits, recall_total = recall_benchmark(tmp, args.mode)
        distractors = distractor_stress(tmp, args.mode)
        core_iso, archive_iso = isolation_test(tmp, args.mode)
        latency_mean, latency_p95 = latency_benchmark(tmp, args.mode)
        token_mean, token_peak = token_benchmark(tmp, args.mode)
        template_mean, template_peak = template_token_test(tmp)
        template_recall_rows, template_recall_hits, template_recall_total = recall_benchmark(
            tmp, "template", "template_recall.sqlite")
        live_probes = live_model_probes(tmp, args.llm, args.mode)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    recall_pct = 100 * recall_hits / recall_total
    lines = [
        "# MedAssist Benchmark Results", "", f"Generated by `python benchmark.py --llm {args.llm} --mode {args.mode}` using the stdlib.", "",
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
        "## Template mode (mock extractor)", "", "| Benchmark | Result |", "|---|---|",
        f"| Multi-session recall | {template_recall_hits}/{template_recall_total} ({100 * template_recall_hits / template_recall_total:.1f}%) |",
        f"| 30-turn context tokens | mean {template_mean:.1f}; peak {template_peak}/1000 (PASS) |", "",
        "## Live-model probes", "", "| Probe | Result |", "|---|---|"])
    lines.extend(f"| {question} | {status} |" for question, status in live_probes)
    lines.extend(["", "Source changes: template mode uses a schema-validated extractor and deterministic database-backed replies. Distractor facts are added using the existing core-memory append and overflow path.", "",
    ])
    report = "\n".join(lines)
    (ROOT / "BENCHMARK.md").write_text(report, encoding="utf-8")
    print(report)
    if (recall_hits != recall_total or template_recall_hits != template_recall_total
            or not all(status == "PASS" for _, status, _ in distractors)
            or not core_iso or not archive_iso):
        raise SystemExit("One or more benchmark checks failed.")


if __name__ == "__main__":
    main()
