# DAPE — Domain-Adaptive PEFT Engine

**Zero-cost FastAPI documentation assistant** — Fine-tune open models locally with LoRA (PEFT) on FastAPI documentation.

## 🎯 Overview

DAPE is a production-ready pipeline that:
- Fetches FastAPI documentation (MIT licensed)
- Builds instruction datasets from markdown
- Fine-tunes open models with Parameter-Efficient Fine-Tuning (LoRA)
- Evaluates model performance
- Serves via FastAPI

**No paid APIs or services required** — Everything runs locally.

## 📋 Requirements

- Python 3.9+
- 8GB+ RAM (16GB+ recommended)
- Optional: CUDA-capable GPU (CPU fallback supported)
- ~5GB disk space for models and data

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Fetch FastAPI Documentation

```bash
python scripts/fetch_docs.py
```

This clones the FastAPI repository (shallow) and copies markdown docs to `data/raw/docs/`.

### 3. Build Instruction Dataset

```bash
python scripts/build_dataset.py
```

Generates `data/dataset.jsonl` with 1500-3000 instruction examples.

### 4. Create Benchmark Set

```bash
python scripts/make_benchmark.py
```

Creates `data/benchmark.jsonl` with 100 challenging questions.

### 5. Fine-tune Model

```bash
python training/train_lora.py \
    --model_name_or_path microsoft/DialoGPT-small \
    --dataset_path data/dataset.jsonl \
    --output_dir runs/lora_adapter \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --lora_r 16 \
    --lora_alpha 32
```

**Note:** For CPU-only training, use `--device_map cpu`. Training will be slower but functional.

**Model Options:**
- `microsoft/DialoGPT-small` (117M params) — Fast, good for testing
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (1.1B params) — Better quality
- `TheBloke/tiny-llama` — Alternative tiny model

### 6. Evaluate

```bash
python evaluation/eval.py \
    --base_model microsoft/DialoGPT-small \
    --adapter_dir runs/lora_adapter \
    --benchmark_path data/benchmark.jsonl
```

Results saved to `evaluation/results.json`.

### 7. Run Inference (CLI)

```bash
python training/infer.py \
    --model_dir runs/lora_adapter \
    --prompt "How do I create a FastAPI endpoint?" \
    --max_new_tokens 128
```

### 8. Start FastAPI Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Test the API:**
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "How do I use FastAPI dependencies?", "max_new_tokens": 256}'
```

## 📁 Project Architecture

```
dape-fastapi/
├── app/
│   ├── main.py          # FastAPI server
│   └── schema.py        # Pydantic request/response models
├── configs/
│   └── training_config.yaml  # Training hyperparameters
├── data/
│   ├── raw/
│   │   ├── fastapi-repo/     # Cloned FastAPI repo
│   │   └── docs/              # Extracted markdown docs
│   ├── dataset.jsonl          # Instruction dataset
│   └── benchmark.jsonl        # Evaluation questions
├── evaluation/
│   ├── eval.py          # Evaluation script
│   └── results.json     # Evaluation metrics
├── runs/
│   └── lora_adapter/     # Saved LoRA adapter weights
├── scripts/
│   ├── fetch_docs.py     # Clone and extract docs
│   ├── build_dataset.py  # Build instruction dataset
│   └── make_benchmark.py # Create benchmark set
├── training/
│   ├── train_lora.py     # LoRA fine-tuning script
│   └── infer.py          # CLI inference script
├── requirements.txt
└── README.md
```

## 🔧 Configuration

Edit `configs/training_config.yaml` to adjust training hyperparameters:

```yaml
model_name_or_path: "microsoft/DialoGPT-small"
dataset_path: "data/dataset.jsonl"
output_dir: "runs/lora_adapter"
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
num_train_epochs: 3
learning_rate: 2e-4
lora_r: 16
lora_alpha: 32
lora_dropout: 0.1
bf16: false  # Set to true if GPU supports bf16
fp16: false  # Alternative to bf16
```

## 💻 Hardware Notes

### CPU Fallback
- All scripts support CPU execution
- Training will be slower (expect 10-100x slower than GPU)
- Use `--device_map cpu` flag
- Reduce `per_device_train_batch_size` to 1-2 for CPU

### GPU Recommendations
- **Minimum:** 4GB VRAM (use `--fp16` or `--bf16`)
- **Recommended:** 8GB+ VRAM
- **Optimal:** 16GB+ VRAM (larger batch sizes, faster training)

### Memory Optimization
- Use gradient accumulation: `--gradient_accumulation_steps 4`
- Enable quantization: `bitsandbytes` (8-bit) if available
- Reduce `lora_r` (e.g., 8 instead of 16) for lower memory

## 📊 Dataset Format

**Instruction Dataset (`data/dataset.jsonl`):**
```json
{"instruction": "How do I create a FastAPI endpoint?", "input": "", "output": "Use the @app.get() decorator..."}
```

**Benchmark (`data/benchmark.jsonl`):**
```json
{"id": "b001", "question": "How do I handle file uploads in FastAPI?", "reference": ""}
```

## 🧪 Evaluation Metrics

- **Exact Match:** String equality (where applicable)
- **Keyword Overlap:** Jaccard similarity over token sets
- **Length Difference:** Average difference in response length
- **Semantic Similarity:** Cosine similarity (if `sentence-transformers` installed)

## 🔒 Security & Privacy

- **No external API calls** — All processing is local
- **No data collection** — FastAPI docs are public (MIT licensed)
- **No secrets** — Scripts filter out potential secrets from dataset

## 🐛 Troubleshooting

**Out of Memory:**
- Reduce `per_device_train_batch_size`
- Increase `gradient_accumulation_steps`
- Use `--fp16` or `--bf16`
- Reduce `lora_r` to 8

**Slow Training:**
- Use GPU if available
- Reduce `num_train_epochs` for testing
- Use smaller model (e.g., DialoGPT-small)

**Import Errors:**
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt --upgrade`

## 📝 License

This project uses FastAPI documentation (MIT licensed). All code in this repository is provided as-is for educational and research purposes.

## 🤝 Contributing

Contributions welcome! Please ensure:
- All dependencies remain free/open-source
- Code runs on CPU (with GPU optional)
- No external API dependencies

## 📚 References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PEFT Library](https://github.com/huggingface/peft)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
