"""
FastAPI server for serving fine-tuned model.

Loads model + adapter at startup and provides POST /generate endpoint.

Usage:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

import time
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.schema import GenerateRequest, GenerateResponse

# Global model and tokenizer
model = None
tokenizer = None
model_loaded = False

app = FastAPI(
    title="DAPE - Domain-Adaptive PEFT Engine",
    description="FastAPI documentation assistant powered by fine-tuned language models",
    version="1.0.0"
)


def load_model(adapter_dir: str = "runs/lora_adapter", base_model: str = None):
    """Load model and adapter at startup."""
    global model, tokenizer, model_loaded
    
    adapter_path = Path(adapter_dir)
    
    # Determine base model
    if base_model:
        base_model_name = base_model
    elif adapter_path.exists() and (adapter_path / "adapter_config.json").exists():
        import json
        with open(adapter_path / "adapter_config.json", 'r') as f:
            config = json.load(f)
            base_model_name = config.get("base_model_name_or_path", "microsoft/DialoGPT-small")
    else:
        base_model_name = "microsoft/DialoGPT-small"
    
    print(f"📥 Loading base model: {base_model_name}")
    
    try:
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
        if adapter_path.exists() and (
            (adapter_path / "adapter_model.safetensors").exists() or
            (adapter_path / "adapter_model.bin").exists()
        ):
            print(f"📥 Loading adapter from: {adapter_dir}")
            model = PeftModel.from_pretrained(model, str(adapter_path))
            model = model.merge_and_unload()
            print("✓ Adapter loaded and merged")
        else:
            print("⚠ No adapter found, using base model only")
        
        model.eval()  # Set to evaluation mode
        model_loaded = True
        print("✓ Model loaded successfully")
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    adapter_dir = Path("runs/lora_adapter")
    if adapter_dir.exists():
        load_model(str(adapter_dir))
    else:
        print("⚠ No adapter directory found. Model will not be loaded.")
        print("  Run training first or specify adapter directory.")


def generate_text(prompt: str, max_new_tokens: int = 256) -> str:
    """Generate text from prompt."""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Format prompt
    formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
    
    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
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
    """Root endpoint."""
    return {
        "message": "DAPE - Domain-Adaptive PEFT Engine",
        "status": "running",
        "model_loaded": model_loaded
    }


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
