# Model Card — DAPE FastAPI Assistant (LoRA Adapter)

## Base Model
Qwen/Qwen2.5-0.5B-Instruct

## Adaptation Method
Parameter-Efficient Fine-Tuning (PEFT) using LoRA adapters.

## Training Data
Instruction-style examples generated from FastAPI documentation (MIT licensed), converted into structured JSONL format.

## Objective
Improve model alignment toward:
- FastAPI routing patterns
- Dependency injection patterns
- Request/response models
- Code-style answers

## Evaluation Method
- Base vs Fine-tuned comparison
- Code block presence
- FastAPI keyword coverage
- Question-output semantic similarity

## Limitations
- Smoke training (200 examples) is not sufficient for consistent performance gains.
- Evaluation uses heuristic metrics (no gold reference answers).
- CPU training on macOS is slow; MPS generation can be unstable.

## Intended Use
Educational and demonstration purposes.
Not for production-critical decision making.

## Ethical Notes
Dataset derived from MIT-linsed documentation.
No proprietary data used.
