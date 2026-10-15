"""Rule-based safety classification and help-seeking policy for patient turns."""
import re

LOW = "LOW"
MEDIUM = "MEDIUM"
HIGH = "HIGH"
SELF_HARM = "SELF_HARM"

# Patterns tolerate punctuation, repeated punctuation, and a trailing backslash.
RED_FLAGS = (
    (re.compile(r"\bheart\s+attack\b", re.I), "possible heart attack"),
    (re.compile(r"\bcardiac\s+arrest\b", re.I), "possible cardiac arrest"),
    (re.compile(r"\bstroke\b", re.I), "possible stroke"),
    (re.compile(r"\b(?:can't|cannot|can\s+not)\s+breathe\b|\btrouble\s+breathing\b|\bshortness\s+of\s+breath\b", re.I),
     "breathing difficulty"),
    (re.compile(r"\bchest\s+(?:pain|tightness|pressure)\b", re.I), "chest pain or pressure"),
    (re.compile(r"\b(?:fainted|passed\s+out|unconscious)\b", re.I), "loss of consciousness"),
    (re.compile(r"\bseizure\b", re.I), "possible seizure"),
    (re.compile(r"\bsevere\s+bleeding\b", re.I), "severe bleeding"),
    (re.compile(r"\boverdose\b", re.I), "possible overdose"),
    (re.compile(r"\banaphylaxis\b|\bsevere\s+(?:allergic\s+)?reaction\b|\bswelling\s+of\s+(?:my\s+)?(?:face|throat)\b", re.I),
     "severe allergic reaction"),
    (re.compile(r"\b(?:i\s+am|i'm|im)\s+dying\b|\bgoing\s+to\s+die\b", re.I), "reports feeling close to death"),
    (re.compile(r"\bcall\s+(?:an\s+)?ambulance\b|\bemergency\b", re.I), "emergency help requested"),
)
SELF_HARM_PATTERNS = (
    (re.compile(r"\bkill\s+myself\b", re.I), "self-harm risk: suicidal intent"),
    (re.compile(r"\bwant\s+to\s+die\b", re.I), "self-harm risk: desire to die"),
    (re.compile(r"\bend\s+my\s+life\b", re.I), "self-harm risk: suicidal thoughts"),
    (re.compile(r"\bsuicid(?:e|al|ality)\b", re.I), "self-harm risk: suicidal thoughts"),
    (re.compile(r"\bhurt\s+myself\b", re.I), "self-harm risk: self-injury"),
)
DISTRESS = re.compile(r"\b(?:dying|emergency|severe|unbearable|help\s+me|urgent|worst\s+pain)\b", re.I)
STOP_REQUEST = re.compile(r"\b(?:stop|discontinue|quit|withdraw|no\s+longer\s+take)\b.{0,50}\b(?:my\s+)?(?:medication|medicine|drug|pill|dose|\w+)", re.I)
MED_CHANGE = re.compile(
    r"\b(?:switch(?:ed)?|chang(?:e|ed)|bump(?:ed)?|start(?:ed|ing)?|stop(?:ped|ping)?|discontinu(?:e|ed|ing)|"
    r"increase|decrease|reduce|raise|lower|double|skip|miss)\b.{0,70}\b(?:medication|medicine|drug|pill|dose|mg|mcg|\w+)\b", re.I,
)
VITALS = re.compile(r"\b(?:bp|blood pressure|glucose|blood sugar|weight|heart rate|pulse|temperature)\b", re.I)


def classify_risk(user_message: str, proposed_writes=()):
    """Return (risk, reason), prioritizing safety signals over proposed writes."""
    text = str(user_message or "") + "\n" + "\n".join(str(write) for write in (proposed_writes or ()))
    for pattern, reason in SELF_HARM_PATTERNS:
        if pattern.search(text):
            return SELF_HARM, reason
    for pattern, reason in RED_FLAGS:
        if pattern.search(text):
            return HIGH, reason
    if STOP_REQUEST.search(text):
        return HIGH, "request to stop a medication"
    if MED_CHANGE.search(text):
        return MEDIUM, "medication change"
    if VITALS.search(text):
        return LOW, "vitals or routine note"
    if DISTRESS.search(text):
        return HIGH, "distress language requires urgent human assessment"
    return LOW, "routine note"


def identifiable_symptom(user_message: str):
    """Return a concise identifiable emergency symptom, or None."""
    text = str(user_message or "")
    for pattern, _ in SELF_HARM_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(0).strip(" \\.,!?;:").lower()
    for pattern, reason in RED_FLAGS:
        match = pattern.search(text)
        if match and reason != "emergency help requested":
            return match.group(0).strip(" \\.,!?;:").lower()
    return None


def decide(risk: str) -> str:
    """Choose proceed, save-with-confirmation, or escalate under asymmetric costs."""
    # Wrong action costs 100 and human review costs 5, so seek help when
    # estimated wrong-action probability exceeds 5/100 = 0.05.
    if risk in (HIGH, SELF_HARM):
        return "ESCALATE"
    if risk == MEDIUM:
        return "SAVE_WITH_CONFIRMATION"
    return "PROCEED"
