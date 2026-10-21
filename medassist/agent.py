"""MemGPT-style agent: core memory in context, messages paged to archival when over budget."""
import os
import time
from datetime import date

from . import db
from .archival import Archive
from .extract import make_extractor
from .llm import get_llm
from .memory import DEFAULT_PERSONA, CoreMemory
from .safety import HIGH, MEDIUM, SELF_HARM, classify_risk, decide, identifiable_symptom
from . import templates
from .tokens import est_tokens


class Agent:
    TOKEN_BUDGET = 1000   # hard cap for active context
    EVICT_AT = 800        # start paging above this
    EVICT_TO = 600        # page down to this
    MAX_HEARTBEATS = 4

    def __init__(self, db_path: str, user_id: str, llm=None, mode: str = "agentic"):
        if mode not in ("agentic", "template"):
            raise ValueError("mode must be 'agentic' or 'template'")
        self.conn = db.connect(db_path)
        self.user_id = user_id
        self.llm = llm or get_llm()
        self.mode = mode
        self.extractor = make_extractor(self.llm) if mode == "template" else None
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

    @staticmethod
    def _fact_value(fields, *names):
        for name in names:
            value = fields.get(name)
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    def _template_store_fact(self, item: dict, tool_log: list):
        kind, fields = item["type"], item["fields"]
        today = date.today().isoformat()
        lines = [ln.strip() for ln in self.core.blocks["human_block"].splitlines() if ln.strip()]
        new_line, old_line = "", None
        if kind in ("medication_change", "medication_start", "medication_stop"):
            med = self._fact_value(fields, "medication", "name", "drug", "medication_name")
            old_med = self._fact_value(fields, "old_medication", "from", "old")
            dose = self._fact_value(fields, "dose", "new_dose", "schedule", "frequency")
            if kind == "medication_stop":
                if not med:
                    return ""
                old_line = next((ln for ln in lines if ln.lower().startswith(f"medication: {med}".lower())), None)
                new_line = f"Discontinued: {med} (stopped {today})"
                if old_line:
                    result = self._run_tool("core_memory_replace", {"block": "human_block", "old": old_line, "new": new_line})
                    tool_log.append(f"core_memory_replace(medication stop) -> {result}")
                    if result.startswith("SUCCESS"):
                        self._archive_medication_history(old_line, new_line, tool_log)
                else:
                    result = self._run_tool("core_memory_append", {"block": "human_block", "fact": new_line})
                    tool_log.append(f"core_memory_append(medication stop) -> {result}")
            else:
                if not med and kind == "medication_change" and old_med:
                    old_line = next((ln for ln in lines if ln.lower().startswith(f"medication: {old_med}".lower())), None)
                    med = old_med
                if not med:
                    return ""
                new_line = f"Medication: {med} {dose}".strip()
                if kind == "medication_change":
                    if old_med:
                        old_line = next((ln for ln in lines if ln.lower().startswith(f"medication: {old_med}".lower())), None)
                    if not old_line:
                        old_line = next((ln for ln in lines if ln.lower().startswith(f"medication: {med}".lower())), None)
                if old_line:
                    result = self._run_tool("core_memory_replace", {"block": "human_block", "old": old_line, "new": new_line})
                    tool_log.append(f"core_memory_replace(medication) -> {result}")
                    if result.startswith("SUCCESS") and kind == "medication_change":
                        self._archive_medication_history(old_line, new_line, tool_log)
                else:
                    result = self._run_tool("core_memory_append", {"block": "human_block", "fact": new_line})
                    tool_log.append(f"core_memory_append(medication) -> {result}")
        elif kind == "vital":
            name = self._fact_value(fields, "name", "vital", "type").lower()
            value = self._fact_value(fields, "value", "reading", "measurement")
            unit = self._fact_value(fields, "unit")
            aliases = {
                "bp": ("BP", "mmHg"), "blood pressure": ("BP", "mmHg"),
                "glucose": ("Glucose", "mg/dL"), "blood sugar": ("Glucose", "mg/dL"),
                "weight": ("Weight", "kg"), "hr": ("HR", "bpm"), "heart rate": ("HR", "bpm"),
                "pulse": ("HR", "bpm"),
            }
            vital_name, default_unit = aliases.get(name, ("", ""))
            if not vital_name or not value:
                return ""
            unit = unit or default_unit
            value_with_unit = value if value.lower().endswith(unit.lower()) else f"{value} {unit}".strip()
            prefix = f"Vital {vital_name}:"
            new_line = f"{prefix} {value_with_unit} ({today})"
            old_line = next((ln for ln in lines if ln.lower().startswith(prefix.lower())), None)
            call_name = "core_memory_replace" if old_line else "core_memory_append"
            args = {"block": "human_block", "old": old_line, "new": new_line} if old_line else {"block": "human_block", "fact": new_line}
            result = self._run_tool(call_name, args)
            tool_log.append(f"{call_name}(vital) -> {result}")
        elif kind == "symptom":
            description = self._fact_value(fields, "description", "symptom", "value")
            if not description:
                return ""
            new_line = f"Symptom ({today}): {description}"
            result = self._run_tool("core_memory_append", {"block": "human_block", "fact": new_line})
            tool_log.append(f"core_memory_append(symptom) -> {result}")
        elif kind == "allergy":
            allergen = self._fact_value(fields, "allergen", "allergy", "value", "name")
            if not allergen:
                return ""
            new_line = f"Allergy: {allergen}"
            old_line = next((ln for ln in lines if ln.lower().startswith(f"allergy: {allergen}".lower())), None)
            call_name = "core_memory_replace" if old_line else "core_memory_append"
            args = {"block": "human_block", "old": old_line, "new": new_line} if old_line else {"block": "human_block", "fact": new_line}
            result = self._run_tool(call_name, args)
            tool_log.append(f"{call_name}(allergy) -> {result}")
        elif kind == "adherence":
            description = self._fact_value(fields, "description", "detail", "value")
            medication = self._fact_value(fields, "medication", "drug")
            if not description:
                return ""
            new_line = f"Adherence ({today}): {medication + ': ' if medication else ''}{description}"
            result = self._run_tool("core_memory_append", {"block": "human_block", "fact": new_line})
            tool_log.append(f"core_memory_append(adherence) -> {result}")
        elif kind == "name":
            name = self._fact_value(fields, "value", "name")
            if not name:
                return ""
            new_line = f"Name: {name}"
            old_line = next((ln for ln in lines if ln.lower().startswith("name:")), None)
            call_name = "core_memory_replace" if old_line else "core_memory_append"
            args = {"block": "human_block", "old": old_line, "new": new_line} if old_line else {"block": "human_block", "fact": new_line}
            result = self._run_tool(call_name, args)
            tool_log.append(f"{call_name}(name) -> {result}")
        elif kind == "condition":
            description = self._fact_value(fields, "description", "condition", "value")
            if not description:
                return ""
            new_line = f"Condition: {description}"
            if not any(ln.lower() == new_line.lower() for ln in lines):
                result = self._run_tool("core_memory_append", {"block": "human_block", "fact": new_line})
                tool_log.append(f"core_memory_append(condition) -> {result}")
        else:
            return ""
        return new_line

    def _archive_medication_history(self, old_line, new_line, tool_log):
        result = self._run_tool("archival_insert", {
            "text": f"Medication history ({date.today().isoformat()}): {old_line} -> {new_line}",
            "kind": "medication_history",
        })
        tool_log.append(f"archival_insert(medication_history) -> {result}")

    def _template_question(self, question):
        # Fetch the persisted core block, so reply values always come from SQLite.
        human = db.get_core_memory(self.conn, self.user_id).get("human_block", "")
        lines = [line.strip() for line in human.splitlines() if line.strip()]
        topic = question.get("topic", "other")
        if topic == "medications":
            return templates.medications([line.split(":", 1)[1].strip() for line in lines if line.startswith("Medication:")])
        if topic == "vitals":
            values = []
            for line in lines:
                if line.startswith("Vital "):
                    values.append(line[len("Vital "):].replace(":", " ", 1))
            return templates.vitals(values)
        if topic == "symptoms":
            return templates.symptoms([line.split(":", 1)[1].strip() for line in lines if line.startswith("Symptom ")])
        if topic == "allergies":
            return templates.allergies([line.split(":", 1)[1].strip() for line in lines if line.startswith("Allergy:")])
        if topic == "labs_history":
            hits = self.archive.search(question.get("query", ""), k=3)
            return templates.history(hits[0]["text"] if hits else "")
        if topic == "summary":
            prefixes = ("Name:", "Condition:", "Allergy:", "Medication:", "Discontinued:", "Vital ", "Symptom ", "Adherence ")
            return templates.summary([line for line in lines if line.startswith(prefixes)][:8])
        return templates.generic()

    def _step_template(self, user_text, risk, tool_log):
        extracted = self.extractor.extract(user_text)
        written = []
        for fact in extracted["facts"]:
            proposed = self._template_store_fact(fact, tool_log)
            if proposed:
                written.append(proposed)
        if extracted["intent"] == "greeting" and not written:
            return templates.greeting()
        topic = extracted["question"].get("topic", "other")
        if extracted["intent"] == "question" and topic != "other":
            return self._template_question(extracted["question"])
        if written:
            # Only include candidate lines confirmed in the persisted human block.
            saved_lines = set(db.get_core_memory(self.conn, self.user_id).get("human_block", "").splitlines())
            confirmed = [line for line in written if line in saved_lines]
            summary_parts = []
            for line in confirmed:
                value = line.split(":", 1)[-1].strip()
                if line.startswith("Vital "):
                    value = line[len("Vital "):].replace(":", " ", 1)
                summary_parts.append(value)
            note = "Please confirm this change with your prescriber." if risk == MEDIUM else ""
            return templates.saved("; ".join(summary_parts), note)
        return templates.generic()

    # ---- one patient turn (with heartbeat loop, Session 07A) -------------------------
    def step(self, user_text: str) -> dict:
        db.log_turn(self.conn, self.session_id, self.user_id, "user", user_text)
        tool_log, reply = [], None
        risk, reason = classify_risk(user_text)
        escalated = False

        def escalate(turn_risk, turn_reason):
            nonlocal risk, reason, escalated
            risk, reason, escalated = turn_risk, turn_reason, True
            symptom = identifiable_symptom(user_text)
            if symptom:
                fact = f"Symptom ({date.today().isoformat()}): {symptom}"
                res = self._run_tool("core_memory_append", {"block": "human_block", "fact": fact})
                tool_log.append(f"core_memory_append(emergency symptom) -> {res}")
                db.log_turn(self.conn, self.session_id, self.user_id, "tool", f"core_memory_append -> {res}")
            if turn_risk == SELF_HARM:
                response = templates.escalation(
                    turn_reason, self_harm=True,
                    emergency_number=os.environ.get(
                        'MEDASSIST_EMERGENCY_NUMBER', 'your local emergency number (112 in India)'
                    ),
                )
            else:
                response = templates.escalation(turn_reason)
            db.log_escalation(self.conn, self.user_id, self.session_id, turn_risk, turn_reason)
            return response

        # Run the safety floor before any backend call; proposed writes are checked below too.
        if risk in (HIGH, SELF_HARM):
            reply = escalate(risk, reason)
        if self.mode == "template" and reply is None:
            self._page_out_if_needed()
            reply = self._step_template(user_text, risk, tool_log)
            db.log_turn(self.conn, self.session_id, self.user_id, "assistant", reply)
            self._page_out_if_needed()
            return {"reply": reply, "tools": tool_log, "tokens": self.context_tokens()}
        for _ in range(self.MAX_HEARTBEATS if reply is None else 0):
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
                reply = escalate(HIGH, turn_reason)
                break
            if turn_risk == SELF_HARM:
                reply = escalate(SELF_HARM, turn_reason)
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
