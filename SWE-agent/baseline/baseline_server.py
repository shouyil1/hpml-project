import uvicorn
from fastapi import FastAPI, Request
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import time
import os

app = FastAPI()

# 1. USE THE BASE MODEL
MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"

print(f"Loading {MODEL_ID} with BitsAndBytes (4-bit)...")

# 2. CONFIGURE 4-BIT QUANTIZATION
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Ensure pad token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --- FIX: FORCE GPU ALLOCATION ---
# Instead of "auto", we force everything to GPU 0.
# This prevents the library from mistakenly offloading layers to CPU.
device_map = {"": "cuda:0"}

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map=device_map
)
print(f"Model loaded on device: {model.device}")

@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": "Qwen/Qwen2.5-14B-Instruct-AWQ"}]}

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    data = await request.json()
    messages = data.get("messages", [])
    
    requested_temp = data.get("temperature", 0.7)
    max_tokens = data.get("max_tokens", 1024)
    
    # Handle Temperature=0
    if requested_temp < 1e-5:
        do_sample = False
        temperature = 1.0 
    else:
        do_sample = True
        temperature = requested_temp

    print(f"Received Request: {len(messages)} msgs. Generating max {max_tokens} tokens...")

    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    start_time = time.time()
    print("Starting generation...")
    
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id
    )
    end_time = time.time()
    print("Generation finished!")
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    token_count = len(generated_ids[0])
    duration = end_time - start_time
    tps = token_count / duration
    print(f"STATS: Generated {token_count} tokens in {duration:.2f}s ({tps:.2f} t/s)")

    return {
        "id": "chatcmpl-baseline",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "Qwen/Qwen2.5-14B-Instruct-AWQ",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response_text},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": len(inputs.input_ids[0]),
            "completion_tokens": token_count,
            "total_tokens": len(inputs.input_ids[0]) + token_count
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)