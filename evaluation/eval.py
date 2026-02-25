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


def load_model_and_tokenizer(adapter_dir: str, base_model_name: str, device: str = "cpu"):
    """Load model with adapter."""
    adapter_path = Path(adapter_dir)
    
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
    
    print(f"📥 Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="cpu",
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
    
    model.to(device)
    print(f"✓ Model on device: {device}")
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    """Generate response from model."""
    device = next(model.parameters()).device
    formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
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


FASTAPI_KEYWORDS = [
    "fastapi",
    "FastAPI",
    "@app.get",
    "@router.get",
    "APIRouter",
    "Depends",
    "pydantic",
    "uvicorn",
    "Request",
]


def _contains_code_block(text: str) -> int:
    """Return 1 if the text contains a markdown code block, else 0."""
    return 1 if "```" in text else 0


def _fastapi_keyword_score(text: str) -> float:
    """Compute normalized FastAPI keyword score based on presence of known tokens."""
    if not text:
        return 0.0
    count = sum(1 for kw in FASTAPI_KEYWORDS if kw in text)
    return count / len(FASTAPI_KEYWORDS)


def _cosine_similarity(a, b) -> float:
    """Cosine similarity between two vectors."""
    from numpy import dot
    from numpy.linalg import norm

    denom = (norm(a) * norm(b))
    if denom == 0:
        return 0.0
    return float(dot(a, b) / denom)


def compute_example_metrics(
    question: str,
    base_output: str,
    ft_output: str,
    encoder: "SentenceTransformer | None" = None,
) -> Dict:
    """Compute metrics for a single example comparing base vs fine-tuned outputs."""
    metrics = {
        "contains_code_block": {},
        "fastapi_keyword_score": {},
        "question_relevance": {},
    }

    # Contains code block
    base_cb = _contains_code_block(base_output)
    ft_cb = _contains_code_block(ft_output)
    metrics["contains_code_block"] = {
        "base": base_cb,
        "ft": ft_cb,
        "delta": ft_cb - base_cb,
    }

    # FastAPI keyword score
    base_kw = _fastapi_keyword_score(base_output)
    ft_kw = _fastapi_keyword_score(ft_output)
    metrics["fastapi_keyword_score"] = {
        "base": base_kw,
        "ft": ft_kw,
        "delta": ft_kw - base_kw,
    }

    # Question relevance via semantic similarity (optional)
    base_rel = None
    ft_rel = None
    if encoder is not None and question and (base_output or ft_output):
        try:
            q_emb = encoder.encode([question])[0]
            if base_output:
                base_emb = encoder.encode([base_output])[0]
                base_rel = _cosine_similarity(q_emb, base_emb)
            if ft_output:
                ft_emb = encoder.encode([ft_output])[0]
                ft_rel = _cosine_similarity(q_emb, ft_emb)
        except Exception as e:
            print(f"⚠ Error computing semantic similarity for example: {e}")

    if base_rel is not None or ft_rel is not None:
        # If one side is missing, treat missing as 0.0 for delta computation
        base_val = base_rel if base_rel is not None else 0.0
        ft_val = ft_rel if ft_rel is not None else 0.0
        metrics["question_relevance"] = {
            "base": base_rel,
            "ft": ft_rel,
            "delta": ft_val - base_val,
        }
    else:
        metrics["question_relevance"] = {
            "base": None,
            "ft": None,
            "delta": None,
        }

    return metrics


def aggregate_metrics(examples: List[Dict]) -> Dict:
    """Aggregate metrics over all examples."""
    sums: Dict[str, Dict[str, float]] = {}
    counts: Dict[str, int] = {}

    for ex in examples:
        ex_metrics = ex.get("metrics", {})
        for metric_name, vals in ex_metrics.items():
            if metric_name not in sums:
                sums[metric_name] = {"base": 0.0, "ft": 0.0, "delta": 0.0}
                counts[metric_name] = 0

            base_val = vals.get("base")
            ft_val = vals.get("ft")
            delta_val = vals.get("delta")

            # Only aggregate when base/ft are not None (for relevance metric)
            if base_val is not None and ft_val is not None and delta_val is not None:
                sums[metric_name]["base"] += float(base_val)
                sums[metric_name]["ft"] += float(ft_val)
                sums[metric_name]["delta"] += float(delta_val)
                counts[metric_name] += 1

    averages: Dict[str, Dict[str, float]] = {}
    for metric_name, total_vals in sums.items():
        count = counts.get(metric_name, 0)
        if count == 0:
            continue
        averages[metric_name] = {
            "base": total_vals["base"] / count,
            "ft": total_vals["ft"] / count,
            "delta": total_vals["delta"] / count,
            "count": count,
        }

    return averages


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
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "mps"],
        default="cpu",
        help="Device for evaluation (default: cpu for reliable macOS behavior)"
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
    
    # Load models (base and fine-tuned)
    print("\n📥 Loading models...")
    base_model, tokenizer = load_model_and_tokenizer(args.base_model, args.base_model, args.device)
    ft_model = None
    if args.adapter_dir:
        ft_model, _ = load_model_and_tokenizer(args.adapter_dir, args.base_model, args.device)
    else:
        print("⚠ No adapter_dir provided, fine-tuned outputs will match base model.")
    
    # Optional semantic encoder
    encoder = None
    if HAS_SENTENCE_TRANSFORMERS:
        try:
            print("📥 Loading sentence-transformers encoder (all-MiniLM-L6-v2)...")
            encoder = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            print(f"⚠ Error loading sentence-transformers encoder, skipping semantic relevance: {e}")
            encoder = None
    
    # Generate predictions
    print("\n🤖 Generating predictions (base vs fine-tuned)...")
    examples: List[Dict] = []
    
    for i, item in enumerate(benchmark):
        question = item.get("question", "")
        
        if not question:
            continue
        
        print(f"  [{i+1}/{len(benchmark)}] {question[:60]}...")
        base_output = generate_response(base_model, tokenizer, question, args.max_new_tokens)
        if ft_model is not None:
            ft_output = generate_response(ft_model, tokenizer, question, args.max_new_tokens)
        else:
            ft_output = base_output
        
        ex_metrics = compute_example_metrics(question, base_output, ft_output, encoder)
        examples.append(
            {
                "question": question,
                "base_output": base_output,
                "ft_output": ft_output,
                "metrics": ex_metrics,
            }
        )
    
    # Aggregate metrics
    print("\n📈 Computing aggregate metrics...")
    averages = aggregate_metrics(examples)
    
    # Prepare results payload
    results = {
        "num_examples": len(examples),
        "averages": averages,
        "examples": examples,
    }
    
    # Save results
    output_path = Path(args.output_path)
    ensure_dir(output_path.parent)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print()
    print("=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Examples evaluated: {results['num_examples']}")
    for metric_name, vals in averages.items():
        print(
            f"{metric_name}: "
            f"base={vals['base']:.3f}, ft={vals['ft']:.3f}, delta={vals['delta']:.3f} "
            f"(count={vals['count']})"
        )
    print(f"\nResults saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
