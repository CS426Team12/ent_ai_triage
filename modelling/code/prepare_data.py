"""
Data preparation for LLM finetuning.
Converts ENT datasets (CSV/XLSX) into training format for LLM finetuning.

Output: modelling/data/training_data.jsonl (one example per line)
Format: {"instruction": "...", "input": "...", "output": "..."}
"""

import pandas as pd
import json
import os
from pathlib import Path
from typing import List, Dict


def load_excel_data(filepath: str) -> pd.DataFrame:
    """Load data from Excel file, handling multiple sheets."""
    try:
        xls = pd.ExcelFile(filepath)
        print(f"📄 Found sheets: {xls.sheet_names}")
        
        dfs = []
        for sheet in xls.sheet_names:
            df = pd.read_excel(filepath, sheet_name=sheet)
            print(f"   {sheet}: {len(df)} rows")
            dfs.append(df)
        
        return pd.concat(dfs, ignore_index=True)
    except Exception as e:
        print(f"❌ Error loading {filepath}: {e}")
        return pd.DataFrame()


def load_csv_data(filepath: str) -> pd.DataFrame:
    """Load data from CSV file."""
    try:
        df = pd.read_csv(filepath)
        print(f"📊 {filepath}: {len(df)} rows")
        return df
    except Exception as e:
        print(f"❌ Error loading {filepath}: {e}")
        return pd.DataFrame()


def create_training_example(row: dict, data_type: str = "generic") -> Dict[str, str]:
    """
    Convert a single data row into training format.
    Handles multiple dataset structures:
    - dataset1.csv: Disease classification (symptoms -> diagnosis)
    - Diagnostic errors: Referral appropriateness
    - Knowledge assessment: Learning outcomes
    
    Output format: {"instruction": "...", "input": "...", "output": "..."}
    """
    
    def find_column(df_columns, keywords):
        """Find column matching keywords (case-insensitive)."""
        for col in df_columns:
            col_lower = col.lower()
            if any(kw in col_lower for kw in keywords):
                return col
        return None
    
    # Dataset1: Symptom-based disease classification
    if data_type == "dataset1":
        # dataset1.csv has: Fever, Headache, Cough, Fatigue, Body_Pain, Disease
        symptom_cols = ["Fever", "Headache", "Cough", "Fatigue", "Body_Pain"]
        disease = str(row.get("Disease", "")).strip()
        
        if not disease or disease.lower() in ["unknown", "nan", ""]:
            return None
        
        # Create symptom description and severity counts from numeric values
        symptoms = []
        severe_count = 0
        fever_val = None
        for col in symptom_cols:
            val = row.get(col)
            if pd.notna(val):
                try:
                    v = float(val)
                    severe_count += 1 if v >= 6 else 0
                    if col.lower() == "fever":
                        fever_val = v
                    severity = "mild" if v < 3 else "moderate" if v < 6 else "severe"
                    symptoms.append(f"{col.lower()}: {severity} ({v:.1f})")
                except (TypeError, ValueError):
                    pass
        
        if not symptoms:
            return None
        
        # Urgency from fever + severe count (align with fix_training_data_urgency.py)
        if fever_val is not None and fever_val >= 104.0 and severe_count >= 4:
            urgency = "urgent"
        elif fever_val is not None and (fever_val >= 103.0 and severe_count >= 3 or fever_val >= 102.0 and severe_count >= 2):
            urgency = "semi-urgent"
        else:
            urgency = "routine"
        
        return {
            "instruction": "You are an ENT triage expert. Classify the urgency of this patient as routine, semi-urgent, or urgent based on their symptoms.",
            "input": "Patient presents with:\n" + "\n".join(symptoms),
            "output": urgency
        }
    
    # Diagnostic errors: Referral appropriateness
    elif data_type == "diagnostic_errors":
        reason = str(row.get("reason for referral", "")).strip()
        ent_diagnosis = str(row.get("entdiag1", "")).strip()
        appropriateness = str(row.get("referral appropriateness", "")).strip()
        
        if not reason or not ent_diagnosis or reason.lower() in ["unknown", "nan", ""]:
            return None
        
        return {
            "instruction": "You are an ENT specialist. Assess whether the patient referral is appropriate given the findings.",
            "input": f"Reason for referral: {reason}\nENT Diagnosis: {ent_diagnosis}",
            "output": f"REFERRAL_ASSESSMENT: {appropriateness}\nCLINICAL_FINDING: {ent_diagnosis}"
        }
    
    # Generic/Knowledge: Use any symptom->outcome mapping
    else:
        # Try to find symptom and outcome columns
        input_col = find_column(row.keys(), ["symptom", "complaint", "chief", "description", "reason"])
        output_col = find_column(row.keys(), ["diagnosis", "urgency", "triage", "severity", "label", "disease"])
        
        if not input_col or not output_col:
            return None
        
        input_text = str(row.get(input_col, "")).strip()
        output_text = str(row.get(output_col, "")).strip()
        
        if not input_text or not output_text or output_text.lower() in ["unknown", "nan", ""]:
            return None
        
        return {
            "instruction": "You are an ENT triage expert. Analyze the patient's symptoms and provide clinical assessment.",
            "input": input_text,
            "output": f"ASSESSMENT: {output_text}"
        }


def prepare_training_data(data_dir: str = "modelling/data") -> List[Dict]:
    """
    Load all ENT data files and convert to training format.
    
    Looks for:
    - dataset1.csv (symptom->disease classification)
    - ENT Patients Data.xlsx
    - Diagnostic error files (referral appropriateness)
    - Knowledge assessment files
    """
    
    training_examples = []
    data_path = Path(data_dir)
    
    # Load dataset1.csv (symptom-based classification)
    dataset1_path = data_path / "dataset1.csv"
    if dataset1_path.exists():
        print(f"\n🔄 Processing dataset1.csv (symptom classification)...")
        df = load_csv_data(str(dataset1_path))
        if not df.empty:
            for _, row in df.iterrows():
                example = create_training_example(row.to_dict(), data_type="dataset1")
                if example:
                    training_examples.append(example)
    
    # Load ENT Patients Data.xlsx
    ent_patients_path = data_path / "ENT Patients Data.xlsx"
    if ent_patients_path.exists():
        print(f"\n🔄 Processing ENT Patients Data.xlsx...")
        df = load_excel_data(str(ent_patients_path))
        if not df.empty:
            for _, row in df.iterrows():
                example = create_training_example(row.to_dict(), data_type="generic")
                if example:
                    training_examples.append(example)
    
    # Load diagnostic error files
    for nested_dir in data_path.iterdir():
        if nested_dir.is_dir() and "diagnostic" in nested_dir.name.lower():
            print(f"\n🔄 Exploring {nested_dir.name}/...")
            for xlsx_file in nested_dir.glob("*.xlsx"):
                print(f"   Processing {xlsx_file.name}...")
                df = load_excel_data(str(xlsx_file))
                if not df.empty:
                    for _, row in df.iterrows():
                        example = create_training_example(row.to_dict(), data_type="diagnostic_errors")
                        if example:
                            training_examples.append(example)
    
    # Load other nested files
    for nested_dir in data_path.iterdir():
        if nested_dir.is_dir() and "diagnostic" not in nested_dir.name.lower():
            print(f"\n🔄 Exploring {nested_dir.name}/...")
            for xlsx_file in nested_dir.glob("*.xlsx"):
                print(f"   Processing {xlsx_file.name}...")
                df = load_excel_data(str(xlsx_file))
                if not df.empty:
                    for _, row in df.iterrows():
                        example = create_training_example(row.to_dict(), data_type="generic")
                        if example:
                            training_examples.append(example)
    
    return training_examples


def save_training_data(examples: List[Dict], output_file: str = "modelling/data/training_data.jsonl"):
    """Save training examples in JSONL format (one JSON per line)."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"\n✅ Saved {len(examples)} training examples to {output_file}")
    
    # Show sample
    if examples:
        print(f"\n📝 Sample training example:")
        print(json.dumps(examples[0], indent=2))


if __name__ == "__main__":
    print("🚀 Preparing ENT training data for LLM finetuning...\n")
    
    # Prepare data
    examples = prepare_training_data()
    
    if examples:
        save_training_data(examples)
    else:
        print("❌ No training examples found. Check data format and column names.")
