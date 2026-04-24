import importlib.util
import pickle
from pathlib import Path


_MODEL = None
_HYBRID_TRIAGE = None
_MODEL_PATH = Path(__file__).resolve().parent.parent / "rf_model" / "urgency_rf_model.pkl"
_UTILS_PATH = Path(__file__).resolve().parent.parent / "rf_model" / "rf_triage_utils.py"


def _normalize_urgency(label: str) -> str:
    val = (label or "").strip().lower()
    mapping = {
        "routine": "routine",
        "semi-urgent": "semi-urgent",
        "semi_urgent": "semi-urgent",
        "semi urgent": "semi-urgent",
        "urgent": "urgent",
    }
    return mapping.get(val, "routine")


def _load_rf_utils():
    global _HYBRID_TRIAGE
    if _HYBRID_TRIAGE is not None:
        return _HYBRID_TRIAGE
    spec = importlib.util.spec_from_file_location("rf_triage_utils", str(_UTILS_PATH))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load RF utils module at {_UTILS_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _HYBRID_TRIAGE = getattr(module, "hybrid_triage")
    return _HYBRID_TRIAGE


def _load_rf_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with open(_MODEL_PATH, "rb") as f:
        _MODEL = pickle.load(f)
    return _MODEL


def predict_rf_urgency(transcript: str) -> dict:
    """Run RF hybrid triage and normalize urgency labels."""
    model = _load_rf_model()
    hybrid_triage = _load_rf_utils()
    result = hybrid_triage(model, transcript)
    return {
        "urgency": _normalize_urgency(result.get("urgency", "routine")),
        "confidence": float(result.get("confidence", 0.0)),
        "source": result.get("source", "random_forest"),
        "reason": result.get("reason", ""),
        "features": result.get("features", {}),
    }
