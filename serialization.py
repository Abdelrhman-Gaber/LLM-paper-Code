import random
from config import FRAMINGHAM_FEATURES, TARGET_COL

ALT_NAMES = {
    "age": ["age", "Age", "PatientAge", "AgeYears", "age at visit", "patient age years"],
    "sysBP": ["sysBP", "systolic bp", "bp systolic", "sys blood pressure", "systolic pressure"],
    "totChol": ["totChol", "total cholesterol", "cholesterol total", "chol total", "total chol mg"],
    "diaBP": ["diaBP", "diastolic bp", "bp diastolic", "dia blood pressure"],
    "heartRate": ["heart rate", "HeartRate", "pulse", "patient heart rate"],
    "glucose": ["glucose", "Glucose", "blood sugar", "fasting glucose"],
    "sex": ["sex", "Sex", "patient sex", "gender"],
    "is_smoking": ["isSmoking", "is_smoking", "smoking", "current smoker"],
    "diabetes": ["diabetes", "Diabetes", "diabetes status"],
    "BPMeds": ["bp meds", "blood pressure meds", "BPMeds"],
    "prevalentStroke": ["prevalent stroke", "StrokeHistory", "prevalentStroke"],
    "prevalentHyp": ["prevalent hyp", "HypertensionHistory", "prevalentHyp"],
    "cigsPerDay": ["cigsPerDay", "cigarettes per day", "cigs/day"],
}


def random_alt_name(feature: str) -> str:
    if feature in ALT_NAMES:
        return random.choice(ALT_NAMES[feature])
    return feature


def serialize_row_structured(row) -> str:
    parts = []
    for k, v in row.items():
        if k == TARGET_COL:
            continue
        parts.append(f"{k}: {v}")
    return ", ".join(parts)


def serialize_row_natural(row) -> str:
    parts = []
    for k, v in row.items():
        if k == TARGET_COL:
            continue
        lk = k.lower()
        v_str = str(v)
        if "age" in lk:
            parts.append(f"The patient is {v_str} years old.")
        elif "sex" in lk or "gender" in lk:
            parts.append(f"The patient's sex is {v_str}.")
        elif "smok" in lk:
            try:
                v_int = int(v)
            except Exception:
                v_int = 1 if str(v).lower().startswith("y") else 0
            parts.append(f"The patient is {'a smoker' if v_int == 1 else 'not a smoker'}.")
        elif "chol" in lk:
            parts.append(f"The total cholesterol level is {v_str}.")
        elif "sys" in lk:
            parts.append(f"The systolic blood pressure is {v_str}.")
        elif "dia" in lk:
            parts.append(f"The diastolic blood pressure is {v_str}.")
        elif "bp meds" in lk:
            try:
                v_int = int(v)
            except Exception:
                v_int = 1 if str(v).lower().startswith("y") else 0
            parts.append(f"The patient is {'taking' if v_int == 1 else 'not taking'} blood pressure medication.")
        elif "diabetes" in lk:
            try:
                v_int = int(v)
            except Exception:
                v_int = 1 if str(v).lower().startswith("y") else 0
            parts.append(f"The patient {'has' if v_int == 1 else 'does not have'} diabetes.")
        else:
            parts.append(f"{k} is {v_str}.")
    return " ".join(parts)


def serialize_row_compact(row) -> str:
    parts = []
    for k, v in row.items():
        if k == TARGET_COL:
            continue
        parts.append(f"{k}={v}")
    return "; ".join(parts)


def serialize_row(row, fmt: str) -> str:
    if fmt == "structured":
        return serialize_row_structured(row)
    elif fmt == "natural":
        return serialize_row_natural(row)
    elif fmt == "compact":
        return serialize_row_compact(row)
    else:
        raise ValueError(f"Unknown serialization format: {fmt}")
