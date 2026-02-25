---
title: DAPE FastAPI Docs Assistant
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# DAPE — Domain-Adaptive PEFT Engine (FastAPI Docs Assistant)

This repository builds a domain-adapted language model from FastAPI documentation using PEFT (Parameter-Efficient Fine-Tuning) with LoRA. The pipeline fetches public FastAPI docs, builds an instruction dataset, fine-tunes an open base model, evaluates base vs tuned outputs, and serves the model via a local FastAPI API.

The setup is zero-cost: no paid APIs, no paid services, and no external tracking. All training and inference run locally. Apple Silicon is supported; CPU fallback is available for training, evaluation, and API inference.

---

## Project Architecture

End-to-end pipeline:

1. **Fetch docs** — Shallow-clone the FastAPI repo and copy markdown from `docs/` into `data/raw/docs/`.
2. **Build instruction dataset** — Parse markdown into instruction/output pairs and write `data/dataset.jsonl`.
3. **Train LoRA adapter** — Fine-tune the base model with LoRA; save adapter weights under `runs/`.
4. **Evaluate base vs tuned** — Run the same benchmark with base model and base+adapter; compare metrics (code blocks, FastAPI keywords, optional question–output relevance).
5. **Serve model via FastAPI API** — Load base + adapter and expose a `/generate` endpoint for local inference.

---

## Repository Structure

```
dape-fastapi/
├── app/
│   ├── main.py
│   └── schema.py
├── configs/
│   └── training_config.yaml
├── data/
│   ├── raw/
│   │   ├── fastapi-repo/
│   │   └── docs/
│   ├── dataset.jsonl
│   └── benchmark.jsonl
├── evaluation/
│   └── eval.py
├── runs/
│   └── (LoRA adapter checkpoints)
├── scripts/
│   ├── fetch_docs.py
│   ├── build_dataset.py
│   └── make_benchmark.py
├── training/
│   ├── train_lora.py
│   └── infer.py
├── requirements.txt
└── README.md
```

---

## Zero-Cost Guarantee

- No OpenAI or other paid LLM APIs.
- No paid APIs or cloud services for training or inference.
- No paid tracking or analytics services.
- All training, evaluation, and inference run locally on your machine.

---

## Requirements

- Python 3.9+
- 8GB+ RAM (16GB+ recommended for training)
- Optional: CUDA GPU or Apple Silicon (MPS). CPU is supported for all steps.
- Roughly 5GB disk for models and data.

---

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Full Pipeline (from scratch)

Fetch docs, build full dataset, and create benchmark:

```bash
python scripts/fetch_docs.py
python scripts/build_dataset.py
python scripts/make_benchmark.py
```

Train (example with CPU fallback):

```bash
python training/train_lora.py \
  --model_name_or_path microsoft/DialoGPT-small \
  --dataset_path data/dataset.jsonl \
  --output_dir runs/lora_adapter \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --device_map cpu
```

Evaluate base vs adapter:

```bash
python evaluation/eval.py \
  --base_model microsoft/DialoGPT-small \
  --adapter_dir runs/lora_adapter \
  --benchmark_path data/benchmark.jsonl \
  --output_path evaluation/results.json \
  --device cpu
```

Serve API (default CPU):

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## Quick Reproducible Smoke Run

Minimal path to train a small LoRA adapter, evaluate it, and run the API on port 8001.

1. Ensure the full instruction dataset exists, then create a 200-example subset:

```bash
head -200 data/dataset.jsonl > data/dataset_200.jsonl
```

If `data/dataset.jsonl` does not exist yet, run `scripts/fetch_docs.py` and `scripts/build_dataset.py` first.

2. Train LoRA on Qwen 0.5B (CPU example; use `--device_map auto` if you have a GPU):

```bash
python training/train_lora.py \
  --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
  --dataset_path data/dataset_200.jsonl \
  --output_dir runs/qwen_lora_smoke \
  --num_train_epochs 2 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --device_map cpu
```

3. Run evaluation (base vs tuned):

```bash
python evaluation/eval.py \
  --base_model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter_dir runs/qwen_lora_smoke \
  --benchmark_path data/benchmark.jsonl \
  --output_path evaluation/results_qwen_smoke.json \
  --max_new_tokens 140 \
  --device cpu
```

4. Start the API on port 8001 (serve the smoke adapter by pointing the app at `runs/qwen_lora_smoke`; see `app/main.py` startup or set adapter path accordingly):

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

The app loads the adapter from `runs/lora_adapter` by default. For the smoke run either copy/link `runs/qwen_lora_smoke` to `runs/lora_adapter` or change the adapter path in the app startup logic so the server uses `runs/qwen_lora_smoke`.

---

## Evaluation

Benchmark entries have empty `reference` fields. Evaluation does not compare against gold answers; it compares **base model** vs **base + LoRA adapter** on the same questions.

Reported metrics include:

- **contains_code_block** — Whether the output contains a markdown code block (` ``` `).
- **fastapi_keyword_score** — Normalized count of FastAPI-related keywords present in the output.
- **question_relevance** — Cosine similarity between question and output embeddings (optional; requires `sentence-transformers`).

Results JSON includes per-example `base_output`, `ft_output`, and metric deltas, plus overall averages.

Smoke runs use a small dataset (e.g. 200 examples) and few epochs; metrics may not improve noticeably over the base model. Full datasets and longer training are needed for clearer gains.

---

## macOS Troubleshooting

**MPS placeholder storage error**

On macOS, inference or evaluation can hit: `RuntimeError: Placeholder storage has not been allocated on MPS device!` This comes from MPS backend behavior with certain ops.

**Recommendation:** Use CPU for inference, evaluation, and the API server. Scripts default to `--device cpu`; the API uses CPU unless `DAPE_DEVICE=mps` is set. For training you can try `--device_map mps` on Apple Silicon, but if you see MPS errors, switch to `--device_map cpu`.

---

## Configuration

Training hyperparameters (including LoRA r/alpha, batch size, epochs) can be set in `configs/training_config.yaml` or overridden via CLI in `training/train_lora.py`.

---

## License and References

FastAPI documentation is MIT licensed. This project is for educational and research use.

- [FastAPI](https://fastapi.tiangolo.com/)
- [PEFT](https://github.com/huggingface/peft)
- [Transformers](https://huggingface.co/docs/transformers)
