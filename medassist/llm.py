"""LLM backends. MockLLM is a deterministic rule-based 'model' so the demo runs offline (like the
course notebooks' dual-execution mode). AnthropicLLM is an optional live backend.

Protocol: complete(system, messages) -> {"tools":[{"tool":..,"args":{..}}], "say": str, "heartbeat": bool}
The mock only reads `system` (which contains core memory) and `messages` - never the database - so
recall after a cold boot proves the core-memory hydration works.
"""
import json
import os
import re
from datetime import date

NUM = r"\d+(?:\.\d+)?"
DOSE = rf"(?P<dose>{NUM}\s*(?:mg|mcg|g|units?|iu|ml)\b[^.,;!?\n]*)"
DRUG = r"(?P<drug>[A-Za-z][A-Za-z\-]+)"
SWITCH_RE = re.compile(
    rf"(?:switch(?:ed)?|chang(?:ed)?|mov(?:ed)?)\s+(?:me\s+)?from\s+(?P<old>[A-Za-z][A-Za-z\-]+)[^.;!?\n]*?\s+to\s+{DRUG}\s+{DOSE}",
    re.I,
)
START_RE = re.compile(
    rf"(?:started|starting|now\s+(?:taking|on)|prescribed|put\s+me\s+on|beginning)\s+(?:me\s+)?(?:on\s+)?{DRUG}\s+{DOSE}", re.I
)
STOP_RE = re.compile(r"(?:stopped|discontinued|no\s+longer)\s+(?:taking\s+)?(?P<drug>[A-Za-z][A-Za-z\-]+)", re.I)
SYMPTOM_RE = re.compile(
    r"\b(?:i\s+have|i've\s+got|i'm\s+having|i\s+am\s+having|experiencing|suffering\s+from|i\s+feel|i've\s+been\s+feeling|i'm\s+feeling)\s+(?P<s>[^.!?\n,;]+)",
    re.I,
)
SYMPTOM_WORDS = re.compile(
    r"headache|dizz|nausea|vomit|cough|fever|pain|fatigue|tired|rash|swell|breath|chest|palpitat|blurred|numb|cramp|diarrhea|thirst",
    re.I,
)
URGENT = re.compile(r"chest|breath", re.I)
WH = ("what", "which", "when", "do i", "am i", "how", "can you tell", "show", "list")


def _human_block(system: str) -> str:
    m = re.search(r"\[human_block\]\n(.*?)\n=== END CORE MEMORY ===", system, re.S)
    return m.group(1) if m else ""


def _upsert(lines, prefix, new_line):
    for ln in lines:
        if ln.lower().startswith(prefix.lower()):
            if ln == new_line:
                return None
            return {"tool": "core_memory_replace", "args": {"block": "human_block", "old": ln, "new": new_line}}
    return {"tool": "core_memory_append", "args": {"block": "human_block", "fact": new_line}}


def _find(lines, prefix):
    return next((ln for ln in lines if ln.lower().startswith(prefix.lower())), None)


def _trim(s: str) -> str:
    return re.split(r"\s+(?:and|but|because)\s+", s.strip())[0].strip()


class MockLLM:
    name = "mock-rule-based"

    def complete(self, system: str, messages: list) -> dict:
        human = _human_block(system)
        last = messages[-1] if messages else {"role": "user", "content": ""}
        if last["role"] == "tool":
            return self._after_tool(last["content"])
        return self._on_user(last["content"], human)

    # -- reacting to tool output (heartbeat continuation) ---------------------------
    def _after_tool(self, content: str) -> dict:
        name, _, result = content.partition(" -> ")
        if result.startswith(("ERROR", "REJECTED")):
            return {"say": f"I couldn't save that update ({result[:140]})"}
        if name.startswith("archival_search"):
            hits = re.findall(r"^\[(\d+)\]\s+(.*)$", result, re.M)
            if not hits:
                return {"say": "I couldn't find anything matching that in your archived records."}
            texts = sorted((h[1] for h in hits), key=lambda t: t[:10], reverse=True)[:2]
            return {"say": "From your archived records (most recent first): " + " | ".join(t[:220] for t in texts)}
        return {"say": "Done."}

    # -- reacting to a patient message ---------------------------------------------
    def _on_user(self, text: str, human: str) -> dict:
        t, low = text.strip(), text.strip().lower()
        if "?" in t or low.startswith(WH):
            return self._answer(low, t, human)
        lines = [ln.strip() for ln in human.splitlines() if ln.strip()]
        today = date.today().isoformat()
        calls, notes = [], []

        def add(call, note):
            if call:
                calls.append(call)
                notes.append(note)

        m = re.search(r"my name is\s+([A-Za-z]+)", t, re.I)
        if m:
            add(_upsert(lines, "Name:", f"Name: {m.group(1).capitalize()}"), f"name = {m.group(1).capitalize()}")

        sw = SWITCH_RE.search(t)
        if sw:
            old, drug, dose = sw["old"].capitalize(), sw["drug"].capitalize(), _trim(sw["dose"])
            new_line = f"Medication: {drug} {dose}"
            line = _find(lines, f"Medication: {old}")
            call = (
                {"tool": "core_memory_replace", "args": {"block": "human_block", "old": line, "new": new_line}}
                if line
                else {"tool": "core_memory_append", "args": {"block": "human_block", "fact": new_line}}
            )
            add(call, f"medication {old} -> {drug} {dose}")
        else:
            st = START_RE.search(t)
            if st:
                drug, dose = st["drug"].capitalize(), _trim(st["dose"])
                add(_upsert(lines, f"Medication: {drug}", f"Medication: {drug} {dose}"), f"medication {drug} {dose}")
        for sp in STOP_RE.finditer(t):
            drug = sp["drug"].capitalize()
            line = _find(lines, f"Medication: {drug}")
            if line:
                add(
                    {"tool": "core_memory_replace", "args": {"block": "human_block", "old": line,
                                                             "new": f"Discontinued: {drug} (stopped {today})"}},
                    f"{drug} discontinued",
                )

        al = re.search(r"allergic to\s+([A-Za-z\-]+)", t, re.I)
        if al:
            a = al.group(1).capitalize()
            add(_upsert(lines, f"Allergy: {a}", f"Allergy: {a}"), f"allergy {a}")

        bp = re.search(r"(?:\bbp\b|blood pressure)\D{0,15}(\d{2,3})\s*/\s*(\d{2,3})", t, re.I)
        if bp:
            v = f"{bp.group(1)}/{bp.group(2)} mmHg"
            add(_upsert(lines, "Vital BP:", f"Vital BP: {v} ({today})"), f"BP {v}")
        gl = re.search(r"(?:blood sugar|glucose)\D{0,15}(\d{2,3})", t, re.I)
        if gl:
            add(_upsert(lines, "Vital Glucose:", f"Vital Glucose: {gl.group(1)} mg/dL ({today})"), f"glucose {gl.group(1)} mg/dL")
        wt = re.search(rf"weigh\w*\D{{0,12}}({NUM})\s*kg", t, re.I)
        if wt:
            add(_upsert(lines, "Vital Weight:", f"Vital Weight: {wt.group(1)} kg ({today})"), f"weight {wt.group(1)} kg")
        hr = re.search(r"(?:heart rate|pulse)\D{0,12}(\d{2,3})", t, re.I)
        if hr:
            add(_upsert(lines, "Vital HR:", f"Vital HR: {hr.group(1)} bpm ({today})"), f"HR {hr.group(1)} bpm")

        urgent = False
        sy = SYMPTOM_RE.search(t)
        if sy and SYMPTOM_WORDS.search(sy["s"]):
            symptoms = [part.strip() for part in re.split(r"\s+and\s+", sy["s"].strip())]
            symptoms = [s for s in symptoms if SYMPTOM_WORDS.search(s)]
            for s in symptoms:
                add({"tool": "core_memory_append", "args": {"block": "human_block", "fact": f"Symptom ({today}): {s}"}},
                    f"symptom '{s}'")
                urgent = urgent or bool(URGENT.search(s))

        if not calls:
            return {"say": "Understood. Nothing new to save to your record from that. Tell me about medication "
                           "changes, symptoms or vitals and I'll keep it updated."}
        say = "Got it, I've updated your record: " + "; ".join(notes) + "."
        if urgent:
            say += " Chest pain or trouble breathing can be an emergency, please seek urgent care now."
        return {"tools": calls, "say": say, "heartbeat": False}

    def _answer(self, low: str, raw: str, human: str) -> dict:
        lines = [ln.strip() for ln in human.splitlines() if ln.strip()]

        def pick(prefix):
            return [ln[len(prefix):].strip() for ln in lines if ln.lower().startswith(prefix.lower())]

        if "allerg" in low:
            a = pick("Allergy:")
            return {"say": "Recorded allergies: " + ", ".join(a) + "." if a else "No allergies are recorded."}
        if re.search(r"stop|discontin|no longer", low):
            stopped = pick("Discontinued:")
            if not stopped:
                stopped = [line for line in pick("Discontinued:")]
            return {"say": "Previously discontinued medications: " + "; ".join(stopped) + "." if stopped else "No discontinued medications are recorded."}
        if re.search(r"medic|taking|prescri|drug|pill|dose", low):
            m = pick("Medication:")
            return {"say": "Your current medications: " + "; ".join(m) + "." if m else "No medications are recorded."}
        if re.search(r"\bname\b|who am i", low):
            names = pick("Name:")
            return {"say": "Your name is " + "; ".join(names) + "." if names else "I don't have your name recorded."}
        if re.search(r"\bbp\b|blood pressure|vital|glucose|blood sugar|weight|heart rate|pulse", low):
            v = []
            for ln in lines:
                if not ln.startswith("Vital "):
                    continue
                fact = ln[len("Vital "):]
                m = re.match(r"(BP|Glucose|Weight|HR):?\s*(.*?)\s*\((\d{4}-\d{2}-\d{2})\)$", fact)
                v.append(f"{m.group(1)} {m.group(2)} ({m.group(3)})" if m else fact)
            return {"say": "Latest vitals on file: " + "; ".join(v) + "." if v else "No recent vitals are recorded."}
        if re.search(r"symptom|feeling", low):
            s = []
            for ln in lines:
                m = re.match(r"Symptom\s*\((\d{4}-\d{2}-\d{2})\):\s*(.*)", ln)
                if m:
                    s.append(f"{m.group(2)} (reported {m.group(1)})")
            return {"say": "Symptoms on file: " + "; ".join(s) + "." if s else "No active symptoms are recorded."}
        if re.search(r"lab|hba1c|a1c|cholesterol|ldl|creatinine|result|visit|history|previous|test|report", low):
            return {"tools": [{"tool": "archival_search", "args": {"query": raw, "k": 3}}], "heartbeat": True}
        return {"say": "I track medications, vitals, symptoms and history. Ask about any of them, or tell me what changed."}


class AnthropicLLM:
    """Optional live backend. Set ANTHROPIC_API_KEY and MEDASSIST_LLM=anthropic."""

    name = "anthropic"

    def __init__(self):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = os.environ.get("MEDASSIST_MODEL", "claude-sonnet-5-5")

    def complete(self, system: str, messages: list) -> dict:
        msgs = []
        for m in messages:
            role = "assistant" if m["role"] == "assistant" else "user"
            content = m["content"] if m["role"] != "tool" else f"[tool result] {m['content']}"
            if msgs and msgs[-1]["role"] == role:
                msgs[-1]["content"] += "\n" + content
            else:
                msgs.append({"role": role, "content": content})
        if not msgs or msgs[0]["role"] != "user":
            msgs.insert(0, {"role": "user", "content": "(session start)"})
        resp = self.client.messages.create(model=self.model, max_tokens=400, system=system, messages=msgs)
        text = resp.content[0].text
        try:
            return json.loads(text[text.index("{"): text.rindex("}") + 1])
        except (ValueError, json.JSONDecodeError):
            return {"say": text}


def get_llm():
    if os.environ.get("MEDASSIST_LLM", "mock").lower() == "anthropic":
        return AnthropicLLM()
    return MockLLM()
