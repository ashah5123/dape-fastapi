#!/usr/bin/env python3
"""
Inference script for fine-tuned model.

Loads base model + LoRA adapter and generates text from prompts.

Usage:
    python training/infer.py \
        --model_dir runs/lora_adapter \
        --prompt "How do I create a FastAPI endpoint?" \
        --max_new_tokens 128
"""

import argparse
import csv
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(model_dir: str, base_model_name: str = None, device: str = "cpu"):
    """Load model and tokenizer, with optional LoRA adapter."""
    model_dir_path = Path(model_dir)
    
    # Resolve device: use MPS only if requested and available, else CPU
    if device == "mps":
        mps_available = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        if not mps_available:
            print("⚠ MPS requested but not available, using CPU")
            device = "cpu"
        else:
            device = "mps"
    else:
        device = "cpu"
    
    # Determine base model
    if base_model_name:
        base_model = base_model_name
    else:
        # Try to read from adapter config
        adapter_config_path = model_dir_path / "adapter_config.json"
        if adapter_config_path.exists():
            import json
            with open(adapter_config_path, 'r') as f:
                config = json.load(f)
                base_model = config.get("base_model_name_or_path", "microsoft/DialoGPT-small")
        else:
            base_model = "microsoft/DialoGPT-small"
    
    print(f"📥 Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="cpu",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True
    )
    
    # Load adapter if exists
    if (model_dir_path / "adapter_model.safetensors").exists() or \
       (model_dir_path / "adapter_model.bin").exists():
        print(f"📥 Loading LoRA adapter from: {model_dir}")
        model = PeftModel.from_pretrained(model, model_dir)
        model = model.merge_and_unload()  # Merge adapter for faster inference
        print("✓ Adapter loaded and merged")
    else:
        print("⚠ No adapter found, using base model only")
    
    model.to(device)
    print(f"✓ Model on device: {device}")
    return model, tokenizer


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    """Generate text from prompt."""
    device = next(model.parameters()).device
    # Format prompt
    formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
    
    # Tokenize and move inputs to same device as model
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract response (everything after "### Response:\n")
    if "### Response:\n" in generated_text:
        response = generated_text.split("### Response:\n")[-1].strip()
    else:
        response = generated_text[len(formatted_prompt):].strip()
    
    return response


def main():
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned model")
    
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory containing adapter weights (or base model)"
    )
    
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Base model name (auto-detected if not provided)"
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Input prompt/question"
    )
    
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling"
    )
    
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Optional CSV file to save outputs"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "mps"],
        default="cpu",
        help="Device for inference (default: cpu for reliable macOS behavior)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Model Inference")
    print("=" * 60)
    print()
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_dir, args.base_model, args.device)
    
    # Generate
    print(f"💬 Prompt: {args.prompt}")
    print("🤖 Generating response...")
    print()
    
    response = generate(
        model,
        tokenizer,
        args.prompt,
        args.max_new_tokens,
        args.temperature,
        args.top_p,
    )
    
    print("=" * 60)
    print("Response:")
    print("=" * 60)
    print(response)
    print("=" * 60)
    
    # Save to CSV if requested
    if args.output_csv:
        output_path = Path(args.output_csv)
        ensure_dir(output_path.parent)
        
        file_exists = output_path.exists()
        with open(output_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["prompt", "response", "max_new_tokens"])
            writer.writerow([args.prompt, response, args.max_new_tokens])
        
        print(f"\n✓ Saved to {output_path}")


def ensure_dir(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
