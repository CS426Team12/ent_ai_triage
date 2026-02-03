"""
Finetune LLM with Unsloth for ENT triage.

Supports multiple models and can be deployed to Ollama after training.
Scalable: Can later swap Unsloth for SageMaker/Modal without changing API.

Usage:
    python modelling/code/finetune_unsloth.py \
        --model-name "unsloth/qwen2.5-0.5b-bnb-4bit" \
        --data-file "modelling/data/training_data.jsonl" \
        --output-dir "modelling/model/finetuned-qwen2.5-ent"
"""

import json
import argparse
import os
from pathlib import Path
from typing import List, Dict
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer


# Model registry for easy switching
MODEL_REGISTRY = {
    "qwen2.5-0.5b": "unsloth/qwen2.5-0.5b-bnb-4bit",
    "qwen2-7b": "unsloth/qwen2-7b-bnb-4bit",
    "phi-2": "unsloth/phi-2-bnb-4bit",
    "mistral-7b": "unsloth/mistral-7b-bnb-4bit",
}


def load_training_data(jsonl_file: str) -> Dataset:
    """Load JSONL training data into Hugging Face Dataset."""
    print(f"📚 Loading training data from {jsonl_file}...")
    
    data = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    print(f"✅ Loaded {len(data)} examples")
    
    # Convert to Dataset format
    dataset = Dataset.from_list(data)
    return dataset


def setup_model(model_name: str, max_seq_length: int = 2048):
    """
    Load and configure model with Unsloth.
    Automatically quantizes to 4-bit for memory efficiency.
    """
    print(f"\n🤖 Setting up model: {model_name}")
    print(f"   Max sequence length: {max_seq_length}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=torch.float16,
        load_in_4bit=True,
    )
    
    # Setup for training
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # LoRA rank
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing=True,
        use_rslora=False,
    )
    
    print("✅ Model configured for training")
    return model, tokenizer


def format_prompt(example: Dict) -> Dict:
    """Format training example into instruction-following format."""
    instruction = example["instruction"]
    input_text = example["input"]
    output_text = example["output"]
    
    # Phi-2/Qwen format
    prompt = f"""Instruction: {instruction}

Input: {input_text}

Output: {output_text}"""
    
    return {"text": prompt}


def finetune(
    model_name: str,
    data_file: str,
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
):
    """
    Finetune model on ENT data.
    
    Args:
        model_name: Model from MODEL_REGISTRY or HF hub
        data_file: Path to training_data.jsonl
        output_dir: Where to save finetuned model
        num_epochs: Number of training epochs
        batch_size: Batch size (4-8 recommended for small models)
        learning_rate: Learning rate for training
    """
    
    # Load data
    dataset = load_training_data(data_file)
    dataset = dataset.map(format_prompt, remove_columns=list(dataset.column_names))
    
    # Setup model
    model, tokenizer = setup_model(model_name)
    
    # Training config
    print(f"\n⚙️ Training configuration:")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Output: {output_dir}")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=10,
        optim="adamw_8bit",
        seed=42,
    )
    
    # Train
    print("\n🚀 Starting training...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        args=training_args,
    )
    
    trainer.train()
    
    # Save
    print(f"\n💾 Saving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"✅ Finetuning complete!")
    print(f"\n📝 Next steps:")
    print(f"   1. Convert to Ollama format: python modelling/code/export_to_ollama.py")
    print(f"   2. Update app/config.py with new model name")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="qwen2.5-0.5b", 
                       help=f"Model key from registry or HF model ID. Options: {list(MODEL_REGISTRY.keys())}")
    parser.add_argument("--data-file", default="modelling/data/training_data.jsonl")
    parser.add_argument("--output-dir", default="modelling/model/finetuned-ent-llm")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    
    args = parser.parse_args()
    
    # Resolve model name
    model_id = MODEL_REGISTRY.get(args.model_name, args.model_name)
    
    # Run
    finetune(
        model_name=model_id,
        data_file=args.data_file,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
