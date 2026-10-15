"""Seed a demo patient: starting core profile + older labs/visits in archival storage."""
from . import db
from .archival import Archive
from .memory import DEFAULT_PERSONA

PROFILE = """Patient ID: {uid}
Name: Unknown
Condition: Type 2 diabetes mellitus (dx 2021)
Condition: Hypertension
Allergy: Penicillin
Medication: Metformin 500 mg twice daily
Medication: Lisinopril 10 mg once daily"""

HISTORY = [
    ("visit", "2024-11-14 VISIT: Annual physical with Dr. Rao. BP 142/90. Lisinopril 10 mg started. Diet and exercise advised."),
    ("lab", "2025-03-05 LAB: HbA1c 8.1% (high), fasting glucose 162 mg/dL, LDL cholesterol 148 mg/dL."),
    ("visit", "2025-06-20 VISIT: Endocrinology follow-up. Metformin continued. Foot exam normal. Eye exam referral placed."),
    ("lab", "2025-09-12 LAB: HbA1c 7.4%, fasting glucose 131 mg/dL, LDL cholesterol 131 mg/dL, creatinine 0.9 mg/dL, eGFR 92."),
    ("lab", "2026-02-10 LAB: Lipid panel LDL cholesterol 118 mg/dL, HDL 46 mg/dL, triglycerides 160 mg/dL."),
]


def seed(db_path: str, user_id: str):
    conn = db.connect(db_path)
    if not db.get_core_memory(conn, user_id):
        db.set_core_memory(conn, user_id, "persona_block", DEFAULT_PERSONA)
        db.set_core_memory(conn, user_id, "human_block", PROFILE.format(uid=user_id))
    arch = Archive(conn, user_id)
    if arch.count() == 0:
        for kind, text in HISTORY:
            arch.insert(text, kind)
    conn.close()
