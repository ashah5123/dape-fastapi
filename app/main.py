"""
FastAPI server for serving fine-tuned model.

Loads model + adapter at startup and provides POST /generate endpoint.

Usage:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

import os
import time
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr

from app.schema import GenerateRequest, GenerateResponse
from app.ui import build_ui

# Global model, tokenizer, and device (cpu by default for stable macOS / Spaces)
model = None
tokenizer = None
model_loaded = False
device = "cpu"

# Adapter / base-model configuration (overridable via env on Spaces)
DAPE_ADAPTER_REPO = os.getenv("DAPE_ADAPTER_REPO", "ashah5123/dape-fastapi-adapter")
DAPE_ADAPTER_DIR = os.getenv("DAPE_ADAPTER_DIR", "./adapter")
DAPE_BASE_MODEL = os.getenv("DAPE_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")


def _resolve_device() -> str:
    """Resolve device from DAPE_DEVICE env (default: cpu). MPS only if requested and available."""
    requested = os.environ.get("DAPE_DEVICE", "cpu").lower().strip()
    if requested == "mps":
        mps_available = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        if not mps_available:
            print("⚠ DAPE_DEVICE=mps but MPS not available, using CPU")
            return "cpu"
        return "mps"
    return "cpu"


root_path = os.getenv("ROOT_PATH", "")

app = FastAPI(
    title="DAPE - Domain-Adaptive PEFT Engine",
    description="FastAPI documentation assistant powered by fine-tuned language models",
    version="1.0.0",
    root_path=root_path,
)


def load_model() -> None:
    """Load base model and LoRA adapter (from HF Hub) at startup."""
    global model, tokenizer, model_loaded, device

    device = _resolve_device()
    print(f"✓ Using device: {device}")

    base_model_name = DAPE_BASE_MODEL
    adapter_repo = DAPE_ADAPTER_REPO
    adapter_dir = DAPE_ADAPTER_DIR
    adapter_path = Path(adapter_dir)
    adapter_path.mkdir(parents=True, exist_ok=True)

    print(f"📥 Loading base model: {base_model_name}")
    print(f"📥 Adapter repo: {adapter_repo}")
    print(f"📁 Local adapter dir: {adapter_path}")

    try:
        # Load tokenizer and base model
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="cpu",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )

        # Try to download adapter from HF Hub
        adapter_loaded = False
        try:
            print("📥 Downloading adapter files from Hugging Face Hub...")
            hf_hub_download(
                repo_id=adapter_repo,
                filename="adapter_config.json",
                local_dir=str(adapter_path),
            )
            hf_hub_download(
                repo_id=adapter_repo,
                filename="adapter_model.safetensors",
                local_dir=str(adapter_path),
            )

            if (adapter_path / "adapter_config.json").exists() and (
                (adapter_path / "adapter_model.safetensors").exists()
                or (adapter_path / "adapter_model.bin").exists()
            ):
                print("📥 Loading adapter from local directory...")
                model = PeftModel.from_pretrained(model, str(adapter_path))
                model = model.merge_and_unload()
                adapter_loaded = True
                print("✓ Adapter loaded and merged")
            else:
                print("⚠ Adapter files not found after download; using base model only")
        except Exception as e:
            print(f"⚠ Failed to download or load adapter from HF Hub: {e}")
            print("   Falling back to base model only.")

        model.to(device)
        model.eval()  # Set to evaluation mode
        model_loaded = True
        if adapter_loaded:
            print("✓ Model + adapter loaded successfully")
        else:
            print("✓ Base model loaded successfully (no adapter)")

    except Exception as e:
        print(f"✗ Error loading model: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Load model on startup (HF Spaces-friendly)."""
    print("🚀 Startup event: loading model...")
    load_model()


def generate_text(prompt: str, max_new_tokens: int = 256) -> str:
    """Generate text from prompt."""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Format prompt
    messages = [
    {"role": "system", "content": "You are a helpful FastAPI documentation assistant. Answer with concise, correct Python code examples when useful."},
    {"role": "user", "content": prompt},
]

formatted_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(formatted_prompt, return_tensors="pt")

    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        repetition_penalty=1.1,
        no_repeat_ngram_size=4,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract response
    if "### Response:\n" in generated_text:
        response = generated_text.split("### Response:\n")[-1].strip()
    else:
        response = generated_text[len(formatted_prompt):].strip()
    
    return response


@app.get("/")
async def root():
    """Root endpoint: redirect to the Gradio UI."""
    return RedirectResponse(url="/ui")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model_loaded
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Generate text from prompt.
    
    Simple concurrency-safe inference (processes one request at a time).
    """
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train a model first.")
    
    start_time = time.time()
    
    try:
        output = generate_text(request.prompt, request.max_new_tokens)
        duration = time.time() - start_time
        
        return GenerateResponse(
            output=output,
            duration_s=duration
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")


gradio_app = build_ui(generate_text)
app = gr.mount_gradio_app(app, gradio_app, path="/ui")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
