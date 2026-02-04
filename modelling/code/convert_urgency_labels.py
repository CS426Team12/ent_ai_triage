import json
import random

# Define mapping rules for urgency based on keywords or random assignment for demo
# In production, use clinical rules or a mapping table

def classify_urgency(entry):
    text = (entry.get('input') or '').lower()
    # Simple rules for demonstration
    if any(word in text for word in [
        'stridor', 'difficulty breathing', 'airway', 'altered mental', 'severe pain', 'facial droop', 'trauma', 'bleeding', 'vomiting', 'confusion', 'high fever', 'sudden hearing loss', 'neck stiffness', 'unable to stop', 'critical', 'severe dizziness', 'meningitis', 'maxillofacial', 'acute', 'obstruction', 'labyrinthitis', 'complications']):
        return 'urgent'
    if any(word in text for word in [
        'moderate', 'persistent', 'difficulty swallowing', 'facial pain', 'dental pain', 'hoarseness', 'tonsillitis', 'abscess', 'sinusitis', 'laryngitis', 'semi-urgent', 'semi urgent', 'abscess', 'pharyngitis', 'swelling', 'painful', 'worsening', 'not improving', 'prolonged', 'week', '5 days', '3 days']):
        return 'semi-urgent'
    # Default
    return 'routine'

input_path = 'modelling/data/training_data.jsonl'
output_path = 'modelling/data/training_data_urgency_only.jsonl'

with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
    for line in fin:
        entry = json.loads(line)
        urgency = classify_urgency(entry)
        # Overwrite output to only URGENCY: <label>
        entry['output'] = f'URGENCY: {urgency}'
        fout.write(json.dumps(entry) + '\n')

print(f"Done. Wrote new file: {output_path}")
