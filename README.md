# MemGPT-Backed Medical Patient Longitudinal Assistant

Run the demo: `python demo.py` (Python 3.9+, stdlib only, works offline).

## Architecture (maps to the course notebooks)

| Component | File | Notebook pattern |
|---|---|---|
| SQLite persistence (sessions, messages, core_memory, archival) | `medassist/db.py` | 08A schema |
| Core memory: `persona_block` + `human_block`, `core_memory_append`, `core_memory_replace`, injection guard | `medassist/memory.py` | 07A |
| Token budget watchdog + paging old messages to archival | `medassist/agent.py` | 07A (budget, `page_to_disk`) |
| Archival hybrid search (BM25 + char-trigram vectors + RRF) | `medassist/archival.py` | 06A |
| Heartbeat agent loop, cold-start hydration | `medassist/agent.py` | 07A / 08A (`hydrate_agent`) |
| Mock LLM (offline) and optional Anthropic backend | `medassist/llm.py` | dual execution mode |

## How the demo works
1. Session 1 (process A): patient says their medication changed. The agent calls `core_memory_replace`, which writes to SQLite.
2. Process exits. Session 2 is a brand new process with zero messages in context.
3. It hydrates the core memory from SQLite and answers with the updated medication. Active context stays under 1,000 tokens; old messages are paged to archival and older labs/visits are retrieved on demand with `archival_search`.

## Interactive use
```
python -m medassist.cli --db app.db --user patient_001 --seed --show-core
```

## Live LLM (optional)
```
pip install anthropic
export ANTHROPIC_API_KEY=...
export MEDASSIST_LLM=anthropic      # optional: MEDASSIST_MODEL=claude-sonnet-5-5
```

Note: a demo/educational tool. It does not diagnose and is not a medical device.
