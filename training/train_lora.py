#!/usr/bin/env python3
"""
Fine-tune a language model using LoRA (Low-Rank Adaptation) with PEFT.

This script:
1. Loads a base model from Hugging Face
2. Applies LoRA adapters using PEFT
3. Fine-tunes on instruction dataset
4. Saves adapter weights (not full model)

Supports CPU fallback with gradient accumulation for small GPUs.

Usage:
    python training/train_lora.py \
        --model_name_or_path microsoft/DialoGPT-small \
        --dataset_path data/dataset.jsonl \
        --output_dir runs/lora_adapter \
        --num_train_epochs 3 \
        --per_device_train_batch_size 4 \
        --lora_r 16 \
        --lora_alpha 32
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)


def ensure_dir(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def load_instruction_dataset(dataset_path: str):
    """Load instruction dataset from JSONL file."""
    data = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    print(f"✓ Loaded {len(data)} examples from {dataset_path}")
    return data


def format_instruction(example: dict) -> str:
    """Format instruction example into prompt."""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")
    
    if input_text:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
    
    return prompt


def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize examples for training."""
    prompts = [format_instruction(ex) for ex in examples]
    
    # Tokenize
    tokenized = tokenizer(
        prompts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )
    
    # For causal LM, labels are the same as input_ids
    tokenized["labels"] = tokenized["input_ids"].clone()
    
    return tokenized


def main():
    parser = argparse.ArgumentParser(description="Fine-tune model with LoRA")
    
    # Model arguments
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="microsoft/DialoGPT-small",
        help="Hugging Face model identifier or local path"
    )
    
    # Data arguments
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/dataset.jsonl",
        help="Path to instruction dataset JSONL file"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/lora_adapter",
        help="Directory to save adapter weights"
    )
    
    # Training arguments
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size per device"
    )
    
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of gradient accumulation steps"
    )
    
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    
    # LoRA arguments
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank (r)"
    )
    
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha scaling factor"
    )
    
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="LoRA dropout"
    )
    
    # Device arguments
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Device map (auto, cpu, cuda, etc.)"
    )
    
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16 precision"
    )
    
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use float16 precision"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("LoRA Fine-tuning Script")
    print("=" * 60)
    print()
    
    # Check device
    if args.device_map == "cpu":
        print("⚠ Warning: Training on CPU will be very slow!")
        print("  Consider using a GPU or reducing batch size.")
    elif args.device_map == "auto":
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            args.device_map = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            print("✓ MPS available (Apple Silicon)")
            args.device_map = "mps"
        else:
            print("⚠ No CUDA/MPS device found, falling back to CPU")
            args.device_map = "cpu"
    
    # Ensure output directory
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    
    # Load tokenizer and model
    print(f"📥 Loading model: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    # Add padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map=args.device_map,
        torch_dtype=torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32),
        trust_remote_code=True
    )
    
    print(f"✓ Model loaded: {model.config.name_or_path}")
    
    # Select LoRA target modules based on actual module names
    module_names = {name for name, _ in model.named_modules()}
    target_modules: list[str]
    reason: str

    # 1) LLaMA / Mistral-style q_proj/k_proj/v_proj/o_proj
    if any(name.endswith("q_proj") or ".q_proj" in name for name in module_names):
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        reason = "found q_proj-style attention projection layers"
    # 2) GPT-2-style c_attn / c_proj
    elif any(name.endswith("c_attn") for name in module_names):
        target_modules = ["c_attn", "c_proj"]
        reason = "found GPT-2-style c_attn projection layers"
    # 3) Qwen-style Wqkv / out_proj
    elif any("Wqkv" in name for name in module_names) and any(
        "out_proj" in name for name in module_names
    ):
        target_modules = ["Wqkv", "out_proj"]
        reason = "found Qwen-style Wqkv/out_proj projection layers"
    # 4) Fallback: generic q_proj / v_proj
    else:
        target_modules = ["q_proj", "v_proj"]
        reason = "fallback to generic q_proj/v_proj projections"
    
    print(f"Using LoRA target modules: {target_modules} (reason: {reason})")
    
    # Configure LoRA
    print(f"🔧 Configuring LoRA (r={args.lora_r}, alpha={args.lora_alpha})...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load dataset
    print(f"📊 Loading dataset: {args.dataset_path}")
    dataset = load_instruction_dataset(args.dataset_path)
    
    # Tokenize dataset
    print("🔤 Tokenizing dataset...")
    tokenized_dataset = []
    for example in dataset:
        tokenized = tokenize_function([example], tokenizer, args.max_length)
        tokenized_dataset.append({
            "input_ids": tokenized["input_ids"][0],
            "attention_mask": tokenized["attention_mask"][0],
            "labels": tokenized["labels"][0],
        })
    
    # Convert to dataset format
    from datasets import Dataset
    train_dataset = Dataset.from_list(tokenized_dataset)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        fp16=args.fp16,
        bf16=args.bf16,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        report_to="none",  # No wandb/tensorboard (zero-cost)
        remove_unused_columns=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("🚀 Starting training...")
    print(f"   Epochs: {args.num_train_epochs}")
    print(f"   Batch size: {args.per_device_train_batch_size}")
    print(f"   Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"   Effective batch size: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
    print()
    
    trainer.train()
    
    # Save adapter
    print(f"💾 Saving adapter to {output_dir}...")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    print()
    print("=" * 60)
    print("✓ Training complete!")
    print(f"  Adapter saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
