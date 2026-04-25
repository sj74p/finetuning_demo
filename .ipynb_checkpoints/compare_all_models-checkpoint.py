import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import gc

# Config
BASE_MODEL_ID = "google/gemma-3-270m"
INSTRUCT_MODEL_ID = "google/gemma-3-270m-it"
POOR_ADAPTER = "gemma3-270m-email-lora-adapter"
IMPROVED_ADAPTER = "gemma3-270m-email-lora-adapter-improved"

prompt = "Rewrite professionally: This code is garbage and broke the build."
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

def generate_text(model, tokenizer, is_base=False, improved=False):
    if is_base:
        inputs = tokenizer(f"Request: {prompt}\nResponse:", return_tensors="pt").to(device)
    else:
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)
    
    gen_kwargs = {
        "max_new_tokens": 64,
        "do_sample": True if improved else False,
        "temperature": 0.7 if improved else 1.0,
        "repetition_penalty": 1.1 if improved else 1.0,
    }

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
        
    input_length = inputs["input_ids"].shape[1]
    text = tokenizer.decode(out[0][input_length:], skip_special_tokens=True).strip()
    return text

def main():
    print(f"🚀 ULTIMATE COMPARISON DEMO (Device: {device.upper()})")
    outputs = {}

    # 1. Base Model
    print("\n--- 1. BASE MODEL ---")
    tokenizer_base = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.float32).to(device)
    outputs['Base'] = generate_text(model, tokenizer_base, is_base=True)
    
    # 2. Instruct Model
    print("\n--- 2. INSTRUCT MODEL ---")
    del model
    gc.collect()
    tokenizer_it = AutoTokenizer.from_pretrained(INSTRUCT_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(INSTRUCT_MODEL_ID, torch_dtype=torch.float32).to(device)
    outputs['Instruct'] = generate_text(model, tokenizer_it)
    
    # 3. Poor LoRA
    print("\n--- 3. POOR LoRA (High LR, Low Epochs) ---")
    try:
        model = PeftModel.from_pretrained(model, POOR_ADAPTER, adapter_name="poor")
        outputs['Poor LoRA'] = generate_text(model, tokenizer_it)
    except:
        outputs['Poor LoRA'] = "(Adapter not found)"

    # 4. Improved LoRA
    print("\n--- 4. IMPROVED LoRA (Stable LR, High Epochs) ---")
    try:
        model.load_adapter(IMPROVED_ADAPTER, adapter_name="improved")
        model.set_adapter("improved")
        outputs['Improved LoRA'] = generate_text(model, tokenizer_it, improved=True)
    except:
        outputs['Improved LoRA'] = "(Adapter not found)"

    print("\n" + "="*80)
    print("🏁 FINAL SCOREBOARD")
    print("="*80)
    print(f"PROMPT: {prompt}\n")
    print(f"[ 1. BASE ]\n{outputs['Base']}\n")
    print(f"[ 2. INSTRUCT ]\n{outputs['Instruct']}\n")
    print(f"[ 3. POOR LORA ]\n{outputs['Poor LoRA']}\n")
    print(f"[ 4. IMPROVED LORA ]\n{outputs['Improved LoRA']}\n")
    print("="*80)

if __name__ == "__main__":
    main()
