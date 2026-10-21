"""Short deterministic response templates for template mode."""

MAX_REPLY_WORDS = 79


def _short(text):
    words = str(text).split()
    return " ".join(words[:MAX_REPLY_WORDS])


def saved(summary: str, prescriber_note: str = "") -> str:
    if not summary:
        return _short("I didn't catch anything to save. Please tell me the detail again.")
    note = prescriber_note.strip()
    return _short(f"Saved: {summary}. {note}".strip())


def medications(items) -> str:
    return _short(f"Your current medications: {'; '.join(items)}." if items else "I don't have any current medications in your record.")


def vitals(items) -> str:
    return _short(f"Latest vitals: {'; '.join(items)}." if items else "I don't have vitals in your record.")


def symptoms(items) -> str:
    return _short(f"Symptoms on file: {'; '.join(items)}." if items else "I don't have symptoms in your record.")


def allergies(items) -> str:
    return _short(f"Recorded allergies: {'; '.join(items)}." if items else "I don't have allergies in your record.")


def history(result: str) -> str:
    return _short(f"I found this in your records: {result}" if result else "I couldn't find that in your archived records.")


def summary(items) -> str:
    return _short(f"Summary on file: {'; '.join(items)}." if items else "I don't have a patient summary in your record.")


def greeting() -> str:
    return "Hello. How can I help with your health record today?"


def escalation(reason: str, self_harm: bool = False, emergency_number: str = "your local emergency number (112 in India)") -> str:
    if self_harm:
        return _short(
            "ESCALATED: I'm sorry you're going through this. Please contact emergency services "
            f"({emergency_number}) or a crisis line now, and reach out to someone nearby who can stay with you."
        )
    return _short(f"ESCALATED: {reason.capitalize()}. Please contact a clinician or emergency services now.")


def generic() -> str:
    return "I didn't catch anything to save. You can ask about your record or tell me what changed."
