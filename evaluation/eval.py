#!/usr/bin/env python3
"""
Evaluate fine-tuned model against benchmark questions.

Computes metrics:
- Exact match (where applicable)
- Keyword overlap (Jaccard similarity)
- Average length difference
- Semantic similarity (if sentence-transformers available)

Usage:
    python evaluation/eval.py \
        --base_model microsoft/DialoGPT-small \
        --adapter_dir runs/lora_adapter \
        --benchmark_path data/benchmark.jsonl
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Optional: semantic similarity
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("⚠ sentence-transformers not available, skipping semantic similarity")


def ensure_dir(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def load_benchmark(benchmark_path: str) -> List[Dict]:
    """Load benchmark questions from JSONL."""
    questions = []
    with open(benchmark_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions


def load_model_and_tokenizer(adapter_dir: str, base_model_name: str):
    """Load model with adapter."""
    adapter_path = Path(adapter_dir)
    
    print(f"📥 Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True
    )
    
    # Load adapter if exists
    if (adapter_path / "adapter_model.safetensors").exists() or \
       (adapter_path / "adapter_model.bin").exists():
        print(f"📥 Loading adapter from: {adapter_dir}")
        model = PeftModel.from_pretrained(model, adapter_dir)
        model = model.merge_and_unload()
    else:
        print("⚠ No adapter found, using base model")
    
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    """Generate response from model."""
    formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "### Response:\n" in generated_text:
        response = generated_text.split("### Response:\n")[-1].strip()
    else:
        response = generated_text[len(formatted_prompt):].strip()
    
    return response


def tokenize(text: str) -> set:
    """Tokenize text into set of lowercase tokens."""
    # Simple tokenization: split on whitespace and punctuation
    tokens = re.findall(r'\b\w+\b', text.lower())
    return set(tokens)


def jaccard_similarity(set1: set, set2: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def exact_match(pred: str, ref: str) -> bool:
    """Check if prediction exactly matches reference (case-insensitive)."""
    return pred.strip().lower() == ref.strip().lower()


def compute_metrics(predictions: List[str], references: List[str]) -> Dict:
    """Compute evaluation metrics."""
    metrics = {
        "exact_matches": 0,
        "keyword_overlaps": [],
        "length_differences": [],
        "semantic_similarities": [],
    }
    
    # Exact match
    for pred, ref in zip(predictions, references):
        if ref:  # Only if reference exists
            metrics["exact_matches"] += exact_match(pred, ref)
    
    # Keyword overlap (Jaccard)
    for pred, ref in zip(predictions, references):
        pred_tokens = tokenize(pred)
        ref_tokens = tokenize(ref) if ref else set()
        overlap = jaccard_similarity(pred_tokens, ref_tokens)
        metrics["keyword_overlaps"].append(overlap)
    
    # Length difference
    for pred, ref in zip(predictions, references):
        pred_len = len(pred.split())
        ref_len = len(ref.split()) if ref else 0
        diff = abs(pred_len - ref_len)
        metrics["length_differences"].append(diff)
    
    # Semantic similarity (if available)
    if HAS_SENTENCE_TRANSFORMERS and references:
        print("📊 Computing semantic similarities...")
        try:
            encoder = SentenceTransformer('all-MiniLM-L6-v2')
            pred_embeddings = encoder.encode(predictions)
            ref_embeddings = encoder.encode([r if r else "" for r in references])
            
            from numpy import dot
            from numpy.linalg import norm
            
            for pred_emb, ref_emb in zip(pred_embeddings, ref_embeddings):
                similarity = dot(pred_emb, ref_emb) / (norm(pred_emb) * norm(ref_emb))
                metrics["semantic_similarities"].append(float(similarity))
        except Exception as e:
            print(f"⚠ Error computing semantic similarity: {e}")
    
    # Aggregate metrics
    results = {
        "num_examples": len(predictions),
        "exact_match_rate": metrics["exact_matches"] / len(predictions) if predictions else 0.0,
        "avg_keyword_overlap": sum(metrics["keyword_overlaps"]) / len(metrics["keyword_overlaps"]) if metrics["keyword_overlaps"] else 0.0,
        "avg_length_difference": sum(metrics["length_differences"]) / len(metrics["length_differences"]) if metrics["length_differences"] else 0.0,
    }
    
    if metrics["semantic_similarities"]:
        results["avg_semantic_similarity"] = sum(metrics["semantic_similarities"]) / len(metrics["semantic_similarities"])
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model")
    
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Base model name"
    )
    
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default=None,
        help="Directory containing adapter weights (optional)"
    )
    
    parser.add_argument(
        "--benchmark_path",
        type=str,
        default="data/benchmark.jsonl",
        help="Path to benchmark JSONL file"
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        default="evaluation/results.json",
        help="Path to save evaluation results"
    )
    
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate per question"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Model Evaluation")
    print("=" * 60)
    print()
    
    # Load benchmark
    print(f"📊 Loading benchmark: {args.benchmark_path}")
    benchmark = load_benchmark(args.benchmark_path)
    print(f"✓ Loaded {len(benchmark)} questions")
    
    # Load model
    model_dir = args.adapter_dir if args.adapter_dir else args.base_model
    model, tokenizer = load_model_and_tokenizer(model_dir, args.base_model)
    
    # Generate predictions
    print("\n🤖 Generating predictions...")
    predictions = []
    references = []
    
    for i, item in enumerate(benchmark):
        question = item.get("question", "")
        reference = item.get("reference", "")
        
        if not question:
            continue
        
        print(f"  [{i+1}/{len(benchmark)}] {question[:60]}...")
        response = generate_response(model, tokenizer, question, args.max_new_tokens)
        predictions.append(response)
        references.append(reference)
    
    # Compute metrics
    print("\n📈 Computing metrics...")
    results = compute_metrics(predictions, references)
    
    # Add predictions to results
    results["predictions"] = [
        {"question": item.get("question"), "prediction": pred, "reference": ref}
        for item, pred, ref in zip(benchmark[:len(predictions)], predictions, references)
    ]
    
    # Save results
    output_path = Path(args.output_path)
    ensure_dir(output_path.parent)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print()
    print("=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Examples evaluated: {results['num_examples']}")
    print(f"Exact match rate: {results['exact_match_rate']:.3f}")
    print(f"Avg keyword overlap: {results['avg_keyword_overlap']:.3f}")
    print(f"Avg length difference: {results['avg_length_difference']:.1f}")
    if 'avg_semantic_similarity' in results:
        print(f"Avg semantic similarity: {results['avg_semantic_similarity']:.3f}")
    print(f"\nResults saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
