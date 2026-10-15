"""Rule-based safety classification and help-seeking policy for patient turns."""
import re

LOW = "LOW"
MEDIUM = "MEDIUM"
HIGH = "HIGH"

RED_FLAGS = (
    (re.compile(r"\bchest\s+pain\b", re.I), "red-flag symptom: chest pain"),
    (re.compile(r"\bshortness\s+of\s+breath\b|\btrouble\s+breathing\b|\bcan't\s+breathe\b", re.I),
     "red-flag symptom: breathing difficulty"),
    (re.compile(r"\bsuicid(?:e|al|ality)\b|\bkill\s+myself\b|\bend\s+my\s+life\b", re.I),
     "red-flag symptom: suicidal thoughts"),
    (re.compile(r"\bsevere\s+(?:allergic\s+)?reaction\b|\banaphylaxis\b|\bthroat\s+(?:is\s+)?closing\b", re.I),
     "red-flag symptom: severe allergic reaction"),
)
STOP_REQUEST = re.compile(r"\b(?:stop|discontinue|quit|withdraw|no\s+longer\s+take)\b.{0,50}\b(?:my\s+)?(?:medication|medicine|drug|pill|dose|\w+)", re.I)
MED_CHANGE = re.compile(
    r"\b(?:switch(?:ed)?|chang(?:e|ed)|start(?:ed|ing)?|stop(?:ped|ping)?|discontinu(?:e|ed|ing)|"
    r"increase|decrease|reduce|raise|lower|double|skip|miss)\b.{0,70}\b(?:medication|medicine|drug|pill|dose|mg|mcg|\w+)\b",
    re.I,
)
VITALS = re.compile(r"\b(?:bp|blood pressure|glucose|blood sugar|weight|heart rate|pulse|temperature)\b", re.I)


def classify_risk(user_message: str, proposed_writes=()):
    """Return (risk, reason), prioritizing emergencies over proposed memory writes."""
    text = str(user_message or "") + "\n" + "\n".join(str(write) for write in (proposed_writes or ()))
    for pattern, reason in RED_FLAGS:
        if pattern.search(text):
            return HIGH, reason
    if STOP_REQUEST.search(text):
        return HIGH, "request to stop a medication"
    if MED_CHANGE.search(text):
        return MEDIUM, "medication change"
    if VITALS.search(text):
        return LOW, "vitals or routine note"
    return LOW, "routine note"


def decide(risk: str) -> str:
    """Choose proceed, save-with-confirmation, or escalate under asymmetric costs."""
    # With wrong action cost 100 and asking a human cost 5, ask/escalate whenever
    # estimated wrong-action probability exceeds 5/100 = 0.05. The rule-based
    # HIGH category is treated as certainly above that threshold and always escalates.
    if risk == HIGH:
        return "ESCALATE"
    if risk == MEDIUM:
        return "SAVE_WITH_CONFIRMATION"
    return "PROCEED"
