"""MemGPT-style agent: core memory in context, messages paged to archival when over budget."""
import time

from . import db
from .archival import Archive
from .llm import get_llm
from .memory import DEFAULT_PERSONA, CoreMemory
from .safety import HIGH, MEDIUM, classify_risk, decide
from .tokens import est_tokens


class Agent:
    TOKEN_BUDGET = 1000   # hard cap for active context
    EVICT_AT = 800        # start paging above this
    EVICT_TO = 600        # page down to this
    MAX_HEARTBEATS = 4

    def __init__(self, db_path: str, user_id: str, llm=None):
        self.conn = db.connect(db_path)
        self.user_id = user_id
        self.llm = llm or get_llm()
        self.session_id = f"s_{int(time.time() * 1000)}"
        db.create_session(self.conn, self.session_id, user_id)
        self.archive = Archive(self.conn, user_id)
        self.core = self._hydrate()
        self.paged_total = 0

    # ---- cold-start hydration (Session 08A) ----------------------------------------
    def _hydrate(self) -> CoreMemory:
        blocks = db.get_core_memory(self.conn, self.user_id)
        if not blocks:
            blocks = {"persona_block": DEFAULT_PERSONA, "human_block": f"Patient ID: {self.user_id}"}
        core = CoreMemory(blocks.get("persona_block", DEFAULT_PERSONA), blocks.get("human_block", ""))
        self._persist(core)
        return core

    def _persist(self, core=None):
        core = core or self.core
        for name, content in core.blocks.items():
            db.set_core_memory(self.conn, self.user_id, name, content)

    # ---- context assembly ----------------------------------------------------------
    def build_context(self):
        rows = db.active_messages(self.conn, self.session_id)
        return self.core.compile_prompt(), [{"role": r[1], "content": r[2]} for r in rows]

    def context_tokens(self) -> int:
        system, msgs = self.build_context()
        return est_tokens(system) + sum(est_tokens(m["content"]) + 4 for m in msgs)

    def _page_out_if_needed(self):
        """Evict oldest messages to archival storage until the context is back under EVICT_TO."""
        if self.context_tokens() <= self.EVICT_AT:
            return
        rows = db.active_messages(self.conn, self.session_id)
        tokens, evicted = self.context_tokens(), []
        while tokens > self.EVICT_TO and len(rows) > 1:
            row = rows.pop(0)
            evicted.append(row)
            tokens -= est_tokens(row[2]) + 4
        if evicted:
            blob = " / ".join(f"{r[1]}: {r[2]}" for r in evicted)
            self.archive.insert(f"CHAT {time.strftime('%Y-%m-%d')} (session {self.session_id}): {blob}", "conversation")
            db.mark_paged(self.conn, [r[0] for r in evicted])
            self.paged_total += len(evicted)

    # ---- tools ---------------------------------------------------------------------
    def _run_tool(self, name: str, args: dict) -> str:
        try:
            if name in ("core_memory_append", "core_memory_replace"):
                fn = getattr(self.core, name)
                res = fn(**args)
                if res.startswith("ERROR") and "token limit" in res:
                    evicted = self.core.evict_oldest_symptoms(2)
                    for line in evicted:
                        self.archive.insert(line, "symptom_history")
                    if evicted:
                        res = fn(**args)
                if res.startswith("SUCCESS"):
                    self._persist()
                return res
            if name == "archival_search":
                hits = self.archive.search(
                    args.get("query", ""), int(args.get("k", 3)),
                    bool(args.get("include_conversation", False)),
                )
                if not hits:
                    return "NO MATCH"
                return f"FOUND {len(hits)}:\n" + "\n".join(f"[{h['id']}] {h['text']}" for h in hits)
            if name == "archival_insert":
                return f"SUCCESS: archived as #{self.archive.insert(args['text'], args.get('kind', 'note'))}."
            return f"ERROR: unknown tool '{name}'."
        except Exception as e:  # tool errors go back to the model, never crash the loop
            return f"ERROR: {type(e).__name__}: {e}"

    # ---- one patient turn (with heartbeat loop, Session 07A) -------------------------
    def step(self, user_text: str) -> dict:
        db.log_turn(self.conn, self.session_id, self.user_id, "user", user_text)
        tool_log, reply = [], None
        risk, reason = classify_risk(user_text)
        escalated = False
        for _ in range(self.MAX_HEARTBEATS):
            self._page_out_if_needed()
            system, msgs = self.build_context()
            out = self.llm.complete(system, msgs)
            calls = out.get("tools") or ([{"tool": out["tool"], "args": out.get("args", {})}] if out.get("tool") else [])
            proposed_writes = [
                (c.get("args") or {}).get("fact", (c.get("args") or {}).get("new", ""))
                for c in calls if c.get("tool") in ("core_memory_append", "core_memory_replace")
            ]
            turn_risk, turn_reason = classify_risk(user_text, proposed_writes)
            if turn_risk == HIGH:
                risk, reason, escalated = HIGH, turn_reason, True
                reply = (
                    "ESCALATED: This may need urgent medical attention. Please contact a clinician or "
                    "emergency services now."
                )
                db.log_escalation(self.conn, self.user_id, self.session_id, risk, reason)
                break
            if turn_risk == MEDIUM:
                risk, reason = MEDIUM, turn_reason
            failed = False
            for c in calls:
                res = self._run_tool(c.get("tool"), c.get("args") or {})
                tool_log.append(f"{c.get('tool')}({c.get('args')}) -> {res.splitlines()[0][:90]}")
                db.log_turn(self.conn, self.session_id, self.user_id, "tool", f"{c.get('tool')} -> {res}"[:700])
                failed = failed or res.startswith(("ERROR", "REJECTED"))
            say = out.get("say")
            if say and not failed:
                reply = say
                break
            if not calls and not say:
                reply = "(no response)"
                break
            # otherwise heartbeat: continue so the model can read the tool result
        reply = reply or "(stopped after max heartbeats)"
        if not escalated and decide(risk) == "SAVE_WITH_CONFIRMATION":
            reply += " Please confirm this change with your prescriber."
        db.log_turn(self.conn, self.session_id, self.user_id, "assistant", reply)
        self._page_out_if_needed()
        return {"reply": reply, "tools": tool_log, "tokens": self.context_tokens()}
