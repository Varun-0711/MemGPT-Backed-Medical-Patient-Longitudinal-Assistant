"""Core memory scratchpad (Session 07A): pinned persona_block + human_block with mutation tools."""
import re

from .tokens import est_tokens

INJECTION_RE = re.compile(
    r"system\s+override"
    r"|ignore\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|above|earlier)"
    r"|ignore\s+(?:your\s+)?instructions"
    r"|disregard\s+.{0,30}instructions"
    r"|you\s+are\s+now\b"
    r"|\bevilbot\b|jailbreak|new\s+persona|forget\s+everything",
    re.I,
)

BLOCK_ALIASES = {
    "human": "human_block",
    "human_block": "human_block",
    "persona": "persona_block",
    "persona_block": "persona_block",
}
BLOCK_LIMITS = {"human_block": 350, "persona_block": 150}  # tokens

DEFAULT_PERSONA = (
    "You are MedTrack, a longitudinal clinical-intake assistant. Track the patient's vitals, "
    "symptoms, medications and history accurately, and update core memory whenever facts change. "
    "You do not diagnose or prescribe; advise emergency care for urgent symptoms. Be concise and factual."
)

PROMPT_TEMPLATE = """TOOLS - reply with ONE JSON object: {{"tools":[{{"tool":name,"args":{{...}}}}],"say":"reply to patient","heartbeat":bool}}
- core_memory_append(block, fact): add a line to human_block or persona_block
- core_memory_replace(block, old, new): replace exact text (use for changed medications/vitals)
- archival_search(query, k): search older labs, visits and archived chat
- archival_insert(text, kind): store a note in archival memory
Omit tools to just reply. Set heartbeat=true to continue after a tool result.
=== CORE MEMORY ===
[persona_block]
{persona_block}
[human_block]
{human_block}
=== END CORE MEMORY ==="""


def validate_memory_text(text: str):
    """Return an error string if the text looks like a prompt-injection / memory-poisoning attempt."""
    if INJECTION_RE.search(text or ""):
        return "REJECTED: text matches a prompt-injection pattern and was not written to memory."
    return None


class CoreMemory:
    def __init__(self, persona_block: str, human_block: str):
        self.blocks = {"persona_block": persona_block, "human_block": human_block}

    def _key(self, block: str):
        return BLOCK_ALIASES.get((block or "").strip().lower())

    def compile_prompt(self) -> str:
        return PROMPT_TEMPLATE.format(
            persona_block=self.blocks["persona_block"],
            human_block=self.blocks["human_block"],
        )

    # ---- memory mutation tools -------------------------------------------------
    def core_memory_append(self, block: str, fact: str) -> str:
        key = self._key(block)
        if not key:
            return "ERROR: invalid block. Use 'human_block' or 'persona_block'."
        fact = (fact or "").strip()
        if not fact:
            return "ERROR: empty fact."
        err = validate_memory_text(fact)
        if err:
            return err
        current = self.blocks[key]
        if fact.lower() in [ln.strip().lower() for ln in current.splitlines()]:
            return f"SKIP: already recorded in {key}."
        updated = (current + "\n" + fact).strip()
        if est_tokens(updated) > BLOCK_LIMITS[key]:
            return f"ERROR: {key} would exceed its {BLOCK_LIMITS[key]}-token limit; page older entries to archival."
        self.blocks[key] = updated
        return f"SUCCESS: appended to {key}."

    def core_memory_replace(self, block: str, old: str, new: str) -> str:
        key = self._key(block)
        if not key:
            return "ERROR: invalid block. Use 'human_block' or 'persona_block'."
        err = validate_memory_text(new)
        if err:
            return err
        current = self.blocks[key]
        if old not in current:
            return f"ERROR: text to replace not found in {key}."
        updated = current.replace(old, new, 1)
        if est_tokens(updated) > BLOCK_LIMITS[key]:
            return f"ERROR: {key} would exceed its {BLOCK_LIMITS[key]}-token limit."
        self.blocks[key] = updated
        return f"SUCCESS: replaced text in {key}."

    def evict_oldest_symptoms(self, n: int = 2) -> list:
        """Move the oldest 'Symptom' lines out of the human block (caller archives them)."""
        lines = self.blocks["human_block"].splitlines()
        idx = [i for i, ln in enumerate(lines) if ln.startswith("Symptom")][:n]
        evicted = [lines[i] for i in idx]
        self.blocks["human_block"] = "\n".join(ln for i, ln in enumerate(lines) if i not in set(idx))
        return evicted
