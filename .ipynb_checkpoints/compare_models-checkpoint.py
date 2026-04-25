import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import gc

# 1. Base Model
BASE_MODEL_ID = "google/gemma-3-270m"
# 2. Instruct Model
INSTRUCT_MODEL_ID = "google/gemma-3-270m-it"
# 3. Fine-tuned Model Adapter
ADAPTER_DIR = "gemma3-270m-email-lora-adapter"

prompt = "Rewrite professionally: This code is garbage and broke the build."

print("="*60)
print("PROMPT:", prompt)
print("="*60)

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

def generate_text(model, tokenizer, is_base=False):
    if is_base:
        # Base models usually don't have chat templates or they respond better when just completing the text
        inputs = tokenizer(f"Request: {prompt}\nResponse:", return_tensors="pt").to(device)
    else:
        # Instruct models use chat templates
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to(device)
    
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        
    input_length = inputs["input_ids"].shape[1]
    tokens = out[0][input_length:]
    text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
    return text


def main():
    print(f"Using device: {device.upper()}")
    
    # Track the outputs to print a summary at the end
    output_base = "(Failed to generate)"
    output_it = "(Failed to generate)"
    output_ft = "(Failed to generate)"

    print("\n--- 1. BASE MODEL ---")
    print(f"Loading {BASE_MODEL_ID}...")
    try:
        tokenizer_base = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        model_base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID, torch_dtype=torch.float32
        ).to(device)
        
        output_base = generate_text(model_base, tokenizer_base, is_base=True)
        print("\nRESPONSE:\n" + output_base)
    except Exception as e:
        print(f"Error loading/generating base model: {e}")
    finally:
        # Clear memory before loading the next model
        if 'model_base' in locals():
            del model_base
        if 'tokenizer_base' in locals():
            del tokenizer_base
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

    print("\n--- 2. INSTRUCT MODEL ---")
    print(f"Loading {INSTRUCT_MODEL_ID}...")
    try:
        tokenizer_it = AutoTokenizer.from_pretrained(INSTRUCT_MODEL_ID)
        model_it = AutoModelForCausalLM.from_pretrained(
            INSTRUCT_MODEL_ID, torch_dtype=torch.float32
        ).to(device)

        output_it = generate_text(model_it, tokenizer_it, is_base=False)
        print("\nRESPONSE:\n" + output_it)
    except Exception as e:
        print(f"Error loading instruct model: {e}")
        return # If instruct model fails, fine-tuned will also fail

    print("\n--- 3. FINE-TUNED MODEL (INSTRUCT + LORA) ---")
    print(f"Loading adapter from {ADAPTER_DIR}...")
    try:
        model_ft = PeftModel.from_pretrained(model_it, ADAPTER_DIR)
        model_ft.eval()
        output_ft = generate_text(model_ft, tokenizer_it, is_base=False)
        print("\nRESPONSE:\n" + output_ft)
    except Exception as e:
        print(f"Error loading adapter: {e}")

    print("\n" + "="*80)
    print("FINAL COMPARISON:")
    print("="*80)
    print(f"Prompt: {prompt}\n")
    print(f"[ 1. BASE MODEL ] ({BASE_MODEL_ID})\n{output_base}\n")
    print("-" * 80)
    print(f"[ 2. INSTRUCT MODEL ] ({INSTRUCT_MODEL_ID})\n{output_it}\n")
    print("-" * 80)
    print(f"[ 3. FINE-TUNED ] (Instruct + LoRA)\n{output_ft}\n")
    print("="*80)

if __name__ == "__main__":
    main()
