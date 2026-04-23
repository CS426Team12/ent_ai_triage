"""
rf_triage_utils.py
---------------
Feature extraction, rule-based overrides, and prediction functions

Usage:
    import pickle
    from triage_utils import hybrid_triage

    with open("urgency_rf_model.pkl", "rb") as f:
        model = pickle.load(f)

    result = hybrid_triage(model, transcript)
"""

import re
import pandas as pd
from typing import Dict, List


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_patient_text(transcript: str) -> str:
    """Extract only [Patient] turns and normalize."""
    turns = re.findall(r"\[Patient\]:\s*(.+?)(?=\[AI Bot\]:|$)", transcript, re.DOTALL)
    combined = " ".join(t.strip() for t in turns)
    return normalize_text(combined)


def compile_patterns(patterns: List[str]) -> List[re.Pattern]:
    return [re.compile(p) for p in patterns]


def match_any(patterns: List[re.Pattern], text: str) -> bool:
    return any(p.search(text) for p in patterns)


def is_negated(text: str, keyword: str) -> bool:
    """Simple negation detection within a small window before keyword."""
    pattern = re.compile(rf"(no|not|without)\s+(?:\w+\s+){{0,2}}{keyword}")
    return bool(pattern.search(text))


# ---------------------------------------------------------------------------
# Urgent features
# ---------------------------------------------------------------------------

BREATHING_PATTERNS = compile_patterns([
    r"trouble breathing",
    r"difficulty breathing",
    r"(can.?t|cannot|unable to) breathe",
    r"harder to breathe",
    r"breathing is.{0,20}hard",
    r"noisy.{0,15}breath",
    r"squeaky sound",
    r"shortness of breath",
    r"breath.{0,20}pooling",
    r"breathe.{0,10}mouth",
    r"breathing.{0,15}blocked",
])

def feat_breathing_difficulty(text: str) -> int:
    return int(match_any(BREATHING_PATTERNS, text) and not is_negated(text, "breathing"))


THROAT_SWELLING_PATTERNS = compile_patterns([
    r"throat.{0,20}swelling",
    r"throat is swelling",
    r"swelling up",
    r"neck.{0,15}swelling",
    r"massive swelling",
    r"neck.{0,10}puffy",
    r"throat.{0,15}closing",
    r"tight.{0,10}throat",
    r"swelling under.{0,20}chin",
])

def feat_throat_swelling(text: str) -> int:
    return int(match_any(THROAT_SWELLING_PATTERNS, text))


BLOOD_PATTERN = re.compile(r"\bblood\b|\bbleed\b|\bbleeding\b")

ACTIVE_BLEEDING_PATTERNS = compile_patterns([
    r"right now",
    r"still going",
    r"won'?t stop",
    r"will not stop",
    r"can'?t get it to stop",
    r"still bleeding",
    r"(twenty|thirty|forty|45|30|20)\s*minute",
    r"keeps? coming",
    r"non.?stop",
    r"currently bleeding",
])

RESOLVED_BLEED_PATTERNS = compile_patterns([
    r"(stopped|stop).{0,30}(ago|now|already|earlier|this morning)",
    r"no longer bleeding",
    r"bleed.{0,20}stopped",
    r"it.{0,10}stopped",
    r"stopped on its own",
])

def feat_active_bleeding(text: str) -> int:
    blood    = bool(BLOOD_PATTERN.search(text))
    active   = match_any(ACTIVE_BLEEDING_PATTERNS, text)
    resolved = match_any(RESOLVED_BLEED_PATTERNS, text)
    return int(blood and active and not resolved)

def feat_resolved_bleeding(text: str) -> int:
    blood    = bool(BLOOD_PATTERN.search(text))
    resolved = match_any(RESOLVED_BLEED_PATTERNS, text)
    return int(blood and resolved)


SUDDEN_HEARING_PATTERNS = compile_patterns([
    r"completely lost hearing",
    r"(can.?t|cannot|unable to) hear anything",
    r"total.{0,15}hearing",
    r"hearing.{0,15}gone",
    r"hearing.{0,15}muffled suddenly",
    r"lost.{0,20}hearing.{0,20}(morning|today|suddenly|overnight)",
    r"woke up.{0,40}(can'?t|cannot|couldn'?t).{0,15}hear",
    r"turned off a switch",
    r"both ears.{0,30}(sudden|woke|overnight|this morning)",
])

def feat_sudden_hearing_loss(text: str) -> int:
    return int(match_any(SUDDEN_HEARING_PATTERNS, text))


VERTIGO_PATTERNS = compile_patterns([
    r"room.{0,10}spinning",
    r"spinning.{0,10}room",
    r"spinning badly",
    r"can.{0,5}barely walk",
    r"can.{0,5}barely stand",
    r"can.{0,5}walk.{0,10}straight",
    r"severely.{0,15}dizz",
    r"off balance.{0,15}severe",
])

def feat_severe_vertigo(text: str) -> int:
    spinning    = match_any(VERTIGO_PATTERNS, text)
    vomit_combo = (
        bool(re.search(r"vomit|vomited|threw up", text)) and
        bool(re.search(r"dizz|vertig|spinning|balance", text))
    )
    return int(spinning or vomit_combo)


AIRWAY_OBSTRUCTION_PATTERNS = compile_patterns([
    r"stuck.{0,20}throat",
    r"something.{0,15}(in my |in the )?throat",
    r"lodged.{0,15}throat",
    r"can feel.{0,20}(in my throat|stuck)",
    r"went in wrong",
    r"swallowed.{0,30}(stuck|throat|feel)",
    r"feels? stuck",
    r"bead.{0,20}(nose|nostril)",
    r"put.{0,15}(up|in).{0,10}(nose|nostril)",
])

def feat_airway_obstruction(text: str) -> int:
    return int(match_any(AIRWAY_OBSTRUCTION_PATTERNS, text))


# ---------------------------------------------------------------------------
# Semi-urgent features
# ---------------------------------------------------------------------------

SWALLOW_PATTERNS = compile_patterns([
    r"hard to swallow",
    r"hurts to swallow",
    r"pain.{0,15}swallow",
    r"swallowing.{0,15}(hard|painful|difficult)",
    r"trouble swallowing",
    r"difficulty swallowing",
    r"drooling",
    r"can.{0,5}barely swallow",
    r"saliva.{0,15}(hard|hurt|painful)",
    r"(can't|cannot|unable to|hard to).{0,5}swallow",
])

def feat_swallowing_difficulty(text: str) -> int:
    return int(match_any(SWALLOW_PATTERNS, text))


TEMP_PATTERN = re.compile(r"\b(9[5-9]|10[0-9]|1[01][0-9])(?:\.\d)?\b")

def _extract_max_temp(text: str) -> float:
    temps = TEMP_PATTERN.findall(text)
    return max((float(t) for t in temps), default=0.0)

def feat_high_fever(text: str) -> int:
    return int(_extract_max_temp(text) >= 103.0)

def feat_moderate_fever(text: str) -> int:
    temp = _extract_max_temp(text)
    return int(101.0 <= temp < 103.0)


NECK_LUMP_PATTERNS = compile_patterns([
    r"lump.{0,20}neck",
    r"bump.{0,20}neck",
    r"neck.{0,20}lump",
    r"mass.{0,20}neck",
    r"lump.{0,15}side.{0,15}neck",
    r"lump.{0,15}jaw",
    r"bump.{0,15}throat",
])

def feat_neck_lump(text: str) -> int:
    return int(match_any(NECK_LUMP_PATTERNS, text))


HOARSENESS_PATTERNS = compile_patterns([
    r"hoarse",
    r"losing.{0,10}voice",
    r"voice.{0,15}disappear",
    r"voice.{0,15}different",
    r"voice.{0,15}gone",
    r"can.{0,5}barely.{0,10}whisper",
    r"can.{0,5}barely speak",
])

def feat_prolonged_hoarseness(text: str) -> int:
    """Hoarseness + duration > a few days — red flag for laryngeal pathology."""
    voice_change = match_any(HOARSENESS_PATTERNS, text)
    prolonged    = bool(re.search(r"\b(week|weeks|month|months|\d+ day|\d+ week)\b", text))
    return int(voice_change and prolonged)


RECENT_SURGERY_PATTERNS = compile_patterns([
    r"(just |last |this ).{0,10}(had|had a).{0,15}surgery",
    r"surgery.{0,20}(last week|few days|ago|recently)",
    r"tonsils? out",
    r"sinus surgery",
    r"had.{0,15}(procedure|operation).{0,20}(last|few|this)",
    r"post.?op",
    r"recovering from.{0,15}surgery",
])

def feat_recent_surgery(text: str) -> int:
    """Recent surgical procedure — raises suspicion for post-op complication."""
    return int(match_any(RECENT_SURGERY_PATTERNS, text))


WEIGHT_LOSS_PATTERNS = compile_patterns([
    r"lost.{0,10}\d+.{0,10}pound",
    r"losing weight.{0,20}(without|not trying|can't explain)",
    r"weight loss.{0,20}(without|unintentional|unexplained)",
    r"weight.{0,10}dropped",
    r"\d+.{0,10}pound.{0,20}without trying",
])

def feat_unexplained_weight_loss(text: str) -> int:
    """Unintentional weight loss — systemic red flag."""
    return int(match_any(WEIGHT_LOSS_PATTERNS, text))


EAR_FB_PATTERNS = compile_patterns([
    r"something.{0,20}(in my ear|in the ear|in.{0,5}ear canal)",
    r"(object|bug|bead|button).{0,20}ear",
    r"ear.{0,20}(something|object|stuck|foreign)",
    r"felt.{0,20}(go|went).{0,10}(in|into).{0,10}ear",
    r"went.{0,10}in.{0,10}(my ear|the ear)",
])

def feat_ear_foreign_body(text: str) -> int:
    """Foreign body in the ear canal — needs removal, not typically an emergency."""
    return int(match_any(EAR_FB_PATTERNS, text))


DIZZINESS_PATTERNS = compile_patterns([
    r"\bdizz",
    r"\bvertig",
    r"off balance",
    r"loss of balance",
    r"unsteady",
    r"lightheaded",
    r"spinning",
])

def feat_dizziness(text: str) -> int:
    """Any dizziness/balance disturbance including mild or episodic.
    Severe cases are captured by feat_severe_vertigo; this catches the rest."""
    return int(match_any(DIZZINESS_PATTERNS, text))


# ---------------------------------------------------------------------------
# Context features
# ---------------------------------------------------------------------------

SUDDEN_ONSET_PATTERNS = compile_patterns([
    r"woke up.{0,30}(and|with|to)",
    r"this morning",
    r"tonight",
    r"within the last.{0,20}(hour|day)",
    r"just happened",
    r"all at once",
    r"out of nowhere",
    r"went to bed.{0,20}fine",
    r"started today",
    r"happened.{0,10}today",
    r"came on.{0,15}(fast|quickly|suddenly|rapid)",
    r"overnight",
    r"very suddenly",
])

def feat_sudden_onset(text: str) -> int:
    return int(match_any(SUDDEN_ONSET_PATTERNS, text))


WORSENING_PATTERNS = compile_patterns([
    r"getting worse",
    r"getting harder",
    r"spreading",
    r"more and more",
    r"dramatically worse",
    r"much worse.{0,15}(today|now|this)",
    r"progressively worse",
    r"rapidly.{0,10}(worsen|deteriorat|progress)",
    r"getting.{0,10}(bigger|worse|harder|more)",
    r"(getting|becoming|feels).{0,10}(worse|harder|more painful)",
])

def feat_rapidly_worsening(text: str) -> int:
    return int(match_any(WORSENING_PATTERNS, text))


STABLE_PATTERNS = compile_patterns([
    r"\bstable\b",
    r"same as always",
    r"same old",
    r"no real change",
    r"hasn.t changed",
    r"same.{0,10}last visit",
    r"no change",
    r"hasn.t gotten worse",
    r"about the same",
    r"not getting worse",
])

def feat_stable(text: str) -> int:
    return int(match_any(STABLE_PATTERNS, text))


CHRONIC_PATTERN = re.compile(
    r"\b(months|years|long.?standing|chronic|for years|for a long time"
    r"|always had|my whole life|lifelong|ongoing)\b"
)

def feat_chronic_duration(text: str) -> int:
    return int(bool(CHRONIC_PATTERN.search(text)))


ACUTE_DURATION_PATTERNS = compile_patterns([
    r"\btoday\b",
    r"this morning",
    r"tonight",
    r"within.{0,10}hour",
    r"\bhours? ago\b",
    r"just now",
    r"(one|two|three|a few|several|a couple).{0,10}day",
    r"\d.{0,5}day.{0,5}ago",
    r"since (yesterday|last night|this morning)",
    r"started (yesterday|last night|this morning|today)",
])

def feat_acute_duration(text: str) -> int:
    """Onset within hours to a couple of days — amplifies urgency of clinical features."""
    return int(match_any(ACUTE_DURATION_PATTERNS, text))


# ---------------------------------------------------------------------------
# Routine features
# ---------------------------------------------------------------------------

ROUTINE_PATTERNS = compile_patterns([
    r"\bannual\b",
    r"follow.?up",
    r"check.?up",
    r"due for my",
    r"tune.?up",
    r"\bconsultation\b",
    r"routine",
    r"scheduled.{0,15}(visit|appointment|check)",
    r"regular check",
    r"coming in for.{0,15}(check|visit|follow)",
])

def feat_routine_signal(text: str) -> int:
    return int(match_any(ROUTINE_PATTERNS, text))


ALLERGY_PATTERNS = compile_patterns([
    r"\ballerg",
    r"runny nose",
    r"\bsneez",
    r"itchy eyes",
    r"seasonal",
    r"antihistamine",
    r"loratadine",
    r"zyrtec",
    r"claritin",
    r"hay fever",
    r"post.?nasal drip",
])

def feat_allergies(text: str) -> int:
    return int(match_any(ALLERGY_PATTERNS, text))


REFERRAL_PATTERNS = compile_patterns([
    r"my doctor sent",
    r"referred me",
    r"my primary.{0,10}(said|told|sent|recommended)",
    r"doctor.{0,10}referred",
    r"primary care.{0,10}(said|sent|recommended|suggested)",
    r"doctor.{0,10}said to see",
    r"sent me (here|to you|to this)",
])

def feat_referral(text: str) -> int:
    """Referred by primary care — indicates a planned, non-emergency pathway."""
    return int(match_any(REFERRAL_PATTERNS, text))


GRADUAL_HEARING_PATTERNS = compile_patterns([
    r"gradual.{0,20}hear",
    r"slowly.{0,15}(losing|lost).{0,10}hear",
    r"hearing.{0,20}(over the years|over.*year|gradual|slowly)",
    r"hearing.{0,20}(worse|declining).{0,20}(year|month|gradual)",
    r"(year|years).{0,20}hearing.{0,20}(worse|less|declining)",
])

def feat_gradual_hearing_loss(text: str) -> int:
    """Slow, progressive hearing loss over months/years — routine signal."""
    return int(match_any(GRADUAL_HEARING_PATTERNS, text))


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(transcript: str) -> Dict[str, int]:
    text = extract_patient_text(transcript)

    features = {
        # --- Urgent ---
        "breathing_difficulty":    feat_breathing_difficulty(text),
        "throat_swelling":         feat_throat_swelling(text),
        "active_bleeding":         feat_active_bleeding(text),
        "sudden_hearing_loss":     feat_sudden_hearing_loss(text),
        "severe_vertigo":          feat_severe_vertigo(text),
        "airway_obstruction":      feat_airway_obstruction(text),
        # --- Semi-Urgent ---
        "swallowing_difficulty":   feat_swallowing_difficulty(text),
        "high_fever":              feat_high_fever(text),
        "moderate_fever":          feat_moderate_fever(text),
        "neck_lump":               feat_neck_lump(text),
        "prolonged_hoarseness":    feat_prolonged_hoarseness(text),
        "recent_surgery":          feat_recent_surgery(text),
        "unexplained_weight_loss": feat_unexplained_weight_loss(text),
        "ear_foreign_body":        feat_ear_foreign_body(text),
        "dizziness":               feat_dizziness(text),
        # --- Context ---
        "sudden_onset":            feat_sudden_onset(text),
        "rapidly_worsening":       feat_rapidly_worsening(text),
        "chronic_duration":        feat_chronic_duration(text),
        "stable":                  feat_stable(text),
        "acute_duration":          feat_acute_duration(text),
        # --- Routine ---
        "routine_signal":          feat_routine_signal(text),
        "allergies":               feat_allergies(text),
        "referral":                feat_referral(text),
        "gradual_hearing_loss":    feat_gradual_hearing_loss(text),
    }

    # --- Composite features ---
    features["airway_risk"] = int(
        features["breathing_difficulty"] or
        features["throat_swelling"] or
        features["airway_obstruction"]
    )

    features["infection_risk"] = int(
        features["high_fever"] or
        (features["moderate_fever"] and features["rapidly_worsening"])
    )

    features["subacute_duration"] = int(
        not features["acute_duration"] and
        not features["chronic_duration"]
    )

    features["semi_urgent_signal"] = int(
        (features["swallowing_difficulty"] and not features["airway_risk"]) or
        (features["moderate_fever"] and not features["high_fever"]) or
        (features["neck_lump"] and features["unexplained_weight_loss"]) or
        features["ear_foreign_body"] or
        features["prolonged_hoarseness"]
    )

    return features


# ---------------------------------------------------------------------------
# Rule-based overrides
# ---------------------------------------------------------------------------

def rule_based_override(features: dict) -> str | None:
    """
    Returns urgency label if a hard safety rule fires, otherwise None.
    These rules bypass the RF and always produce 'urgent' at confidence=1.0.
    """
    # Airway emergency (highest priority)
    if features["airway_risk"]:
        return "urgent"

    # Active uncontrolled bleeding
    if features["active_bleeding"]:
        return "urgent"

    # Sudden sensorineural hearing loss
    if features["sudden_hearing_loss"]:
        return "urgent"

    # Severe incapacitating vertigo
    if features["severe_vertigo"]:
        return "urgent"

    return None


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_rf(model, features: dict) -> tuple[str, float]:
    """Returns (predicted_label, confidence)."""
    X_single   = pd.DataFrame([features])
    probs      = model.predict_proba(X_single)[0]
    pred       = model.classes_[probs.argmax()]
    confidence = probs.max()
    return pred, float(confidence)


def hybrid_triage(model, transcript: str) -> dict:
    """
    Main triage pipeline: rule-based override → random forest.

    Returns a dict with:
      urgency    : 'urgent' | 'semi-urgent' | 'routine'
      source     : 'rule' | 'random_forest'
      confidence : float (1.0 for rule overrides)
      reason     : str (only present for rule overrides)
      features   : dict of all feature values
    """
    # Step 1: Extract features
    features = extract_features(transcript)

    # Step 2: Rule-based override
    rule_result = rule_based_override(features)
    if rule_result:
        return {
            "urgency":    rule_result,
            "source":     "rule",
            "confidence": 1.0,
            "reason":     "Critical safety rule triggered",
            "features":   features,
        }

    # Step 3: RF prediction
    rf_pred, rf_conf = predict_rf(model, features)
    return {
        "urgency":    rf_pred,
        "source":     "random_forest",
        "confidence": rf_conf,
        "features":   features,
    }
