"""Structured patient-message extraction for deterministic template mode."""
import re
from datetime import date

from .llm import MockLLM

INTENTS = {"report", "question", "greeting", "other"}
FACT_TYPES = {
    "medication_change", "medication_start", "medication_stop", "vital", "symptom",
    "allergy", "adherence", "name", "condition",
}
QUESTION_TOPICS = {"medications", "vitals", "symptoms", "allergies", "labs_history", "summary", "other"}
MAX_FACTS = 12
MAX_FIELDS = 8
MAX_TEXT = 200

EXTRACTION_SYSTEM = r"""Extract only facts explicitly stated by the patient. Never infer or invent a value.
Return ONE JSON object only, with this exact shape:
{"intent":"report|question|greeting|other","facts":[{"type":"medication_change|medication_start|medication_stop|vital|symptom|allergy|adherence|name|condition","fields":{}}],"question":{"topic":"medications|vitals|symptoms|allergies|labs_history|summary|other","query":"..."}}
Use strings or numbers for field values. Use an empty facts array when no fact was stated. For non-questions use topic other and an empty query.

Examples:
Patient: My doctor changed me from Metformin to Glimepiride 2 mg daily.
JSON: {"intent":"report","facts":[{"type":"medication_change","fields":{"old_medication":"Metformin","medication":"Glimepiride","dose":"2 mg daily"}}],"question":{"topic":"other","query":""}}
Patient: My BP is 120/80 and I have a headache.
JSON: {"intent":"report","facts":[{"type":"vital","fields":{"name":"BP","value":"120/80","unit":"mmHg"}},{"type":"symptom","fields":{"description":"headache"}}],"question":{"topic":"other","query":""}}
Patient: I am allergic to penicillin. What allergies do I have?
JSON: {"intent":"question","facts":[{"type":"allergy","fields":{"allergen":"penicillin"}}],"question":{"topic":"allergies","query":"What allergies do I have?"}}
Patient: What was my last HbA1c?
JSON: {"intent":"question","facts":[],"question":{"topic":"labs_history","query":"What was my last HbA1c?"}}"""


def validate_extraction(value):
    """Validate a decoded extraction object, dropping malformed facts/fields."""
    if not isinstance(value, dict) or value.get("intent") not in INTENTS:
        return None
    facts_in = value.get("facts")
    question_in = value.get("question")
    if not isinstance(facts_in, list) or not isinstance(question_in, dict):
        return None
    topic = question_in.get("topic")
    query = question_in.get("query")
    if topic not in QUESTION_TOPICS or not isinstance(query, str) or len(query) > 500:
        question = {"topic": "other", "query": ""}
    else:
        question = {"topic": topic, "query": query.strip()}
    facts = []
    for item in facts_in[:MAX_FACTS]:
        if not isinstance(item, dict) or item.get("type") not in FACT_TYPES:
            continue
        fields_in = item.get("fields")
        if not isinstance(fields_in, dict) or len(fields_in) > MAX_FIELDS:
            continue
        fields = {}
        valid = True
        for key, raw in fields_in.items():
            if not isinstance(key, str) or not key or len(key) > 40:
                valid = False
                break
            if isinstance(raw, bool) or not isinstance(raw, (str, int, float)):
                valid = False
                break
            text = str(raw).strip()
            if len(text) > MAX_TEXT:
                valid = False
                break
            if text:
                fields[key] = text
        if valid and fields:
            facts.append({"type": item["type"], "fields": fields})
    return {"intent": value["intent"], "facts": facts, "question": question}


def _question_topic(text):
    low = text.lower()
    if re.search(r"\b(?:what is|what's|who is) my name\b", low):
        return "summary"
    if re.search(r"\b(?:which|what) medication did i (?:stop|discontinue)\b", low):
        return "summary"
    if re.search(r"medic|pill|drug|dose|taking", low):
        return "medications"
    if re.search(r"\bbp\b|blood pressure|vital|glucose|blood sugar|weight|pulse|heart rate", low):
        return "vitals"
    if re.search(r"symptom|feeling|how am i", low):
        return "symptoms"
    if "allerg" in low:
        return "allergies"
    if re.search(r"lab|hba1c|a1c|cholesterol|creatinine|visit|history|older|previous|result|test", low):
        return "labs_history"
    if re.search(r"summary|summarize|what do you know|my record", low):
        return "summary"
    return "other"


class RuleBasedExtractor:
    """Offline regex extractor returning the same schema as a live extractor."""

    def extract(self, message: str) -> dict:
        text, low = (message or "").strip(), (message or "").strip().lower()
        facts = []

        def add(kind, **fields):
            if fields:
                facts.append({"type": kind, "fields": fields})

        name = re.search(r"my name is\s+([A-Za-z][A-Za-z '\-]{0,60})", text, re.I)
        if name:
            add("name", value=name.group(1).strip(" .,!?"))

        change = re.search(
            r"(?:switch(?:ed)?|chang(?:ed|e)|bump(?:ed)?|increase(?:d)?|raise(?:d)?)\s+(?:me\s+)?(?:from\s+)?"
            r"(?P<old>[A-Za-z][A-Za-z\-]+)?\s*(?:to|up\s+to)?\s*(?P<med>[A-Za-z][A-Za-z\-]+)?\s*"
            r"(?P<dose>\d+(?:\.\d+)?\s*(?:mg|mcg|g)(?:\s*,?\s*[^.!?;]+)?)", text, re.I)
        if change:
            add("medication_change", old_medication=(change.group("old") or "").strip(),
                medication=(change.group("med") or change.group("old") or "").strip(),
                dose=change.group("dose").strip(" ,"))
        else:
            start = re.search(r"\b(?:started|starting|prescribed|put me on)\s+(?:taking\s+)?(?P<med>[A-Za-z][A-Za-z\-]+)\s+(?P<dose>\d+(?:\.\d+)?\s*(?:mg|mcg|g)\b[^.!?;]*)", text, re.I)
            if start:
                add("medication_start", medication=start.group("med"), dose=start.group("dose").strip())
        stop = re.search(r"\b(?:stopped|discontinued|no longer taking)\s+(?:taking\s+)?(?P<med>[A-Za-z][A-Za-z\-]+)", text, re.I)
        if stop:
            add("medication_stop", medication=stop.group("med"))

        adherence = re.search(r"\b(?:missed|skipped|doubled|out of)\b[^.!?;]{0,150}", text, re.I)
        if adherence:
            add("adherence", description=adherence.group(0).strip())

        allergy = re.search(r"(?:allergic to|allergy to)\s+([A-Za-z][A-Za-z\-]{0,60})", text, re.I)
        if allergy:
            add("allergy", allergen=allergy.group(1))
        bp = re.search(r"(?:\bbp\b|blood pressure)\D{0,15}(\d{2,3})\s*/\s*(\d{2,3})", text, re.I)
        if bp:
            add("vital", name="BP", value=f"{bp.group(1)}/{bp.group(2)}", unit="mmHg")
        glucose = re.search(r"(?:blood sugar|glucose)\D{0,15}(\d{2,3})", text, re.I)
        if glucose:
            add("vital", name="Glucose", value=glucose.group(1), unit="mg/dL")
        weight = re.search(r"weigh\w*\D{0,12}(\d+(?:\.\d+)?)\s*kg", text, re.I)
        if weight:
            add("vital", name="Weight", value=weight.group(1), unit="kg")
        heart = re.search(r"(?:heart rate|pulse)\D{0,12}(\d{2,3})", text, re.I)
        if heart:
            add("vital", name="HR", value=heart.group(1), unit="bpm")

        # Conditions are distinguished from the finite symptom vocabulary.
        condition = re.search(r"\b(?:diagnosed with|have been diagnosed with|i have)\s+(type\s+\d\s+diabetes|diabetes|hypertension|asthma|arthritis)\b", text, re.I)
        if condition:
            add("condition", description=condition.group(1))
        symptoms = re.search(r"\b(?:i have|i've got|i am having|i'm having|experiencing|i feel|i'm feeling)\s+([^.!?\n]+)", text, re.I)
        if symptoms:
            content = symptoms.group(1).strip(" ,;")
            known = re.compile(r"headache|dizz|nausea|vomit|cough|fever|pain|fatigue|tired|rash|swelling|breath|chest|palpitat|blurred|numb|cramp|diarrhea|thirst", re.I)
            for part in re.split(r"\s+and\s+", content):
                if known.search(part):
                    add("symptom", description=part.strip(" ,;"))

        if "?" in text or low.startswith(("what ", "which ", "when ", "how ", "show ", "list ", "can you ")):
            intent, topic, query = "question", _question_topic(text), text[:500]
        elif re.match(r"^(hi|hello|hey|good morning|good afternoon)\b", low):
            intent, topic, query = "greeting", "other", ""
        elif facts:
            intent, topic, query = "report", "other", ""
        else:
            intent, topic, query = "other", "other", ""
        return validate_extraction({
            "intent": intent, "facts": facts,
            "question": {"topic": topic, "query": query},
        })


class MockExtractor(RuleBasedExtractor):
    """Mock backend extractor: deterministic, offline, and schema-compatible."""


class LLMExtractor:
    def __init__(self, llm):
        self.llm = llm
        self.fallback = RuleBasedExtractor()

    def extract(self, message: str) -> dict:
        try:
            candidate = self.llm.complete(EXTRACTION_SYSTEM, [{"role": "user", "content": message}])
            valid = validate_extraction(candidate)
            if valid is not None:
                return valid
        except Exception:
            pass
        return self.fallback.extract(message)


def make_extractor(llm):
    if isinstance(llm, MockLLM):
        return MockExtractor()
    return LLMExtractor(llm)
