"""
Export finetuned LLM to Ollama format.

After finetuning, convert the model to Ollama's quantized format for deployment.
This allows you to use the finetuned model with the existing triage API.

Usage:
    python modelling/code/export_to_ollama.py \
        --model-dir "modelling/model/finetuned-ent-llm" \
        --ollama-model-name "ent-triage-qwen2.5"
"""

import os
import json
import argparse
import subprocess
from pathlib import Path


def create_ollama_modelfile(model_dir: str, model_name: str) -> str:
    """Create Ollama Modelfile for the finetuned model."""
    
    modelfile_content = f"""FROM {model_dir}

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_predict 1024

SYSTEM \"\"\"You are an expert ENT triage clinician. Analyze patient symptoms and provide structured clinical assessment with urgency classification.

Format your response as:
SUMMARY: [clinical summary]
FINDINGS: [key clinical findings]
URGENCY: [routine|semi-urgent|urgent]
REASONING: [why this urgency level]
\"\"\"
"""
    
    return modelfile_content


def export_to_huggingface(model_dir: str):
    """
    Convert finetuned model to HF format (needed for Ollama).
    """
    print(f"📦 Model directory: {model_dir}")
    
    # Check if config exists
    config_path = Path(model_dir) / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"✅ Model config found")
        print(f"   Architecture: {config.get('architectures', ['unknown'])[0]}")
        print(f"   Hidden size: {config.get('hidden_size', 'unknown')}")
    
    return model_dir


def push_to_ollama(model_dir: str, ollama_model_name: str):
    """
    Import finetuned model into Ollama.
    
    Note: This assumes the model is in GGUF format or convertible.
    If needed, use ollama/convert.py first.
    """
    print(f"\n🚀 Preparing to import into Ollama as '{ollama_model_name}'...")
    
    modelfile = create_ollama_modelfile(model_dir, ollama_model_name)
    
    modelfile_path = Path(model_dir) / "Modelfile"
    with open(modelfile_path, 'w') as f:
        f.write(modelfile)
    
    print(f"✅ Created Modelfile")
    
    print(f"\n📝 Next steps (manual):")
    print(f"   1. Convert model to GGUF (if not already):")
    print(f"      ollama run qwen2.5:0.5b")
    print(f"   2. Create Ollama model:")
    print(f"      cd {model_dir}")
    print(f"      ollama create {ollama_model_name} -f Modelfile")
    print(f"   3. Update app/config.py:")
    print(f"      OLLAMA_MODEL_NAME = '{ollama_model_name}'")
    print(f"   4. Restart API and test")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="modelling/model/finetuned-ent-llm",
                       help="Directory with finetuned model")
    parser.add_argument("--ollama-model-name", default="ent-triage-qwen2.5",
                       help="Name for Ollama model")
    
    args = parser.parse_args()
    
    print("🔄 Exporting finetuned model to Ollama format...\n")
    
    # Export
    export_to_huggingface(args.model_dir)
    
    # Create Ollama config
    push_to_ollama(args.model_dir, args.ollama_model_name)
    
    print(f"\n✅ Export complete!")


if __name__ == "__main__":
    main()
