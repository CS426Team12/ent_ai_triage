"""
Simple LLM finetuning using transformers (no Unsloth needed).
Works on Mac CPU.

Usage:
    python modelling/code/finetune_simple.py \
        --model-name "Qwen/Qwen2-0.5B" \
        --data-file "modelling/data/training_data.jsonl" \
        --output-dir "modelling/model/finetuned-ent-llm"
"""

import os
# DISABLE GPU - force CPU only
os.environ["TORCH_DEVICE"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
os.environ["ACCELERATE_CPU_AFFINITY"] = "1"

import json
import argparse
from pathlib import Path
from typing import List, Dict
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)


def load_training_data(jsonl_file: str) -> Dataset:
    """Load JSONL training data into Hugging Face Dataset."""
    print(f"📚 Loading training data from {jsonl_file}...")
    
    data = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    print(f"✅ Loaded {len(data)} examples")
    return Dataset.from_list(data)


def format_prompt(example: Dict) -> Dict:
    """Format training example into instruction-following format."""
    instruction = example["instruction"]
    input_text = example["input"]
    output_text = example["output"]
    
    prompt = f"""Instruction: {instruction}

Input: {input_text}

Output: {output_text}"""
    
    return {"text": prompt}


def finetune(
    model_name: str,
    data_file: str,
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 2e-4,
):
    """
    Finetune model on ENT data using standard transformers.
    
    Args:
        model_name: HF model ID (e.g., "Qwen/Qwen2-0.5B")
        data_file: Path to training_data.jsonl
        output_dir: Where to save finetuned model
        num_epochs: Number of training epochs
        batch_size: Batch size (2-4 recommended for Mac)
        learning_rate: Learning rate for training
    """
    
    # Load data
    print("\n🚀 Loading data...")
    dataset = load_training_data(data_file)
    dataset = dataset.map(format_prompt, remove_columns=list(dataset.column_names))
    
    # Split train/val (90/10)
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    print(f"📊 Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    
    # Load model and tokenizer
    print(f"\n🤖 Loading model: {model_name}")
    
    # Force pure CPU
    device = torch.device("cpu")
    print(f"🖥️  Using device: {device}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model = model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize datasets
    print("🔄 Tokenizing data...")
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            max_length=512,
            truncation=True,
        )
    
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        num_proc=2,
    )
    eval_dataset = eval_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        num_proc=2,
    )
    
    # Training config
    print(f"\n⚙️ Training configuration:")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Output: {output_dir}")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        warmup_steps=50,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=100,
        logging_steps=20,
        save_total_limit=2,
        load_best_model_at_end=True,
        seed=42,
        report_to=[],  # Disable wandb
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    # Train
    print("\n🚀 Starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
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
    parser.add_argument("--model-name", default="Qwen/Qwen2-0.5B",
                       help="HF model ID")
    parser.add_argument("--data-file", default="modelling/data/training_data.jsonl")
    parser.add_argument("--output-dir", default="modelling/model/finetuned-ent-llm")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    
    args = parser.parse_args()
    
    finetune(
        model_name=args.model_name,
        data_file=args.data_file,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
