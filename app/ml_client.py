import joblib
import pandas as pd
from pathlib import Path
from typing import Optional

# Load model at startup
MODEL_PATH = Path(__file__).parent.parent / "modelling" / "model" / "final_ent_triage_model.pkl"
model_data = None

def load_model():
    """Load the trained ML model."""
    global model_data
    if model_data is None and MODEL_PATH.exists():
        try:
            model_data = joblib.load(MODEL_PATH)
            print(f"✓ ML Model loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
    return model_data

def predict_urgency(transcript: str) -> dict:
    """
    Predict urgency from transcript using ML model.
    
    Returns:
        dict with keys:
            - urgency: predicted urgency level (routine, semi-urgent, urgent)
            - confidence: prediction confidence score (0-1)
    """
    model_data = load_model()
    
    if model_data is None:
        return {
            "urgency": "routine",
            "confidence": 0.0
        }
    
    try:
        preprocessor = model_data.get("preprocessor")
        model = model_data.get("model")
        label_mapping = model_data.get("label_mapping")
        
        if not all([preprocessor, model, label_mapping]):
            return {
                "urgency": "routine",
                "confidence": 0.0
            }
        
        # Create a minimal feature dataframe for prediction
        features = extract_features_from_transcript(transcript)
        
        # Create DataFrame with the expected columns
        df = pd.DataFrame([features])
        
        # Preprocess features
        X_transformed = preprocessor.transform(df)
        
        # Make prediction
        prediction_idx = model.predict(X_transformed)[0]
        probabilities = model.predict_proba(X_transformed)[0]
        confidence = float(max(probabilities))
        
        # Reverse map from index to label
        reverse_mapping = {v: k for k, v in label_mapping.items()}
        urgency = reverse_mapping.get(prediction_idx, "routine")
        
        return {
            "urgency": urgency,
            "confidence": confidence
        }
    except Exception as e:
        print(f"❌ ML prediction error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "urgency": "routine",
            "confidence": 0.0
        }



def extract_features_from_transcript(transcript: str) -> dict:
    """
    Extract features from transcript for ML model.
    
    Expected features based on preprocessing.py:
    - categorical: nasal_discharge, language
    - boolean: worsening, fever, dizziness, hearing_change, immunocompromised
    - numeric: duration_days, pain_severity, age
    - text: symptom_keywords
    """
    transcript_lower = transcript.lower()
    
    # Boolean features - detect keywords
    worsening = 1 if any(word in transcript_lower for word in ["worse", "worsening", "getting worse"]) else 0
    fever = 1 if "fever" in transcript_lower else 0
    dizziness = 1 if any(word in transcript_lower for word in ["dizziness", "dizzy", "vertigo"]) else 0
    hearing_change = 1 if any(word in transcript_lower for word in ["hearing", "deaf", "muffled"]) else 0
    immunocompromised = 1 if any(word in transcript_lower for word in ["immunocompromised", "immune", "suppressed", "immunosuppressed"]) else 0
    
    # Numeric features - try to extract from transcript or use defaults
    duration_days = extract_duration(transcript_lower)
    pain_severity = extract_pain_severity(transcript_lower)
    age = extract_age(transcript_lower)
    
    # Categorical features
    nasal_discharge = extract_nasal_discharge_type(transcript_lower)
    language = "english"  # Assume English for now
    
    # Text features
    symptom_keywords = extract_symptom_keywords(transcript_lower)
    
    return {
        "duration_days": duration_days,
        "pain_severity": pain_severity,
        "age": age,
        "nasal_discharge": nasal_discharge,
        "language": language,
        "worsening": worsening,
        "fever": fever,
        "dizziness": dizziness,
        "hearing_change": hearing_change,
        "immunocompromised": immunocompromised,
        "symptom_keywords": symptom_keywords
    }


def extract_duration(text: str) -> int:
    """Extract duration in days from text."""
    import re
    # Look for patterns like "2 days", "1 week", etc.
    days_match = re.search(r'(\d+)\s*days?', text)
    if days_match:
        return int(days_match.group(1))
    
    weeks_match = re.search(r'(\d+)\s*weeks?', text)
    if weeks_match:
        return int(weeks_match.group(1)) * 7
    
    return 2  # Default to 2 days


def extract_pain_severity(text: str) -> int:
    """Extract pain severity (0-10 scale) from text."""
    import re
    # Look for patterns like "3 out of 10"
    severity_match = re.search(r'(\d+)\s*(?:out of|/)\s*10', text)
    if severity_match:
        return int(severity_match.group(1))
    
    # Default severity based on keywords
    if any(word in text for word in ["severe", "unbearable", "excruciating"]):
        return 8
    elif any(word in text for word in ["moderate", "significant"]):
        return 5
    elif any(word in text for word in ["mild", "slight", "annoying"]):
        return 3
    
    return 3  # Default mild


def extract_age(text: str) -> int:
    """Extract age from text."""
    import re
    age_match = re.search(r'\b(\d{1,3})\s*(?:year|yr|yo|old)', text)
    if age_match:
        return int(age_match.group(1))
    
    return 40  # Default age


def extract_nasal_discharge_type(text: str) -> str:
    """Extract type of nasal discharge."""
    if any(word in text for word in ["clear"]):
        return "clear"
    elif any(word in text for word in ["yellow", "greenish", "purulent"]):
        return "yellow"
    elif any(word in text for word in ["blood", "bloody"]):
        return "bloody"
    
    return "clear"  # Default


def extract_symptom_keywords(text: str) -> str:
    """Extract symptom keywords from transcript."""
    symptoms = []
    keywords = {
        "throat_pain": ["sore throat", "throat pain", "throat hurt"],
        "congestion": ["congestion", "stuffy", "blocked"],
        "cough": ["cough", "coughing"],
        "hoarseness": ["hoarse", "voice change"],
        "post_nasal": ["post nasal", "drip"],
        "ear_pain": ["ear pain", "earache"],
    }
    
    for symptom, terms in keywords.items():
        if any(term in text for term in terms):
            symptoms.append(symptom)
    
    return ";".join(symptoms) if symptoms else "general_ent_concern"
