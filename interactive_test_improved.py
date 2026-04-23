import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sys

MODEL_ID = "google/gemma-3-270m-it"
ADAPTER_DIR = "gemma3-270m-email-lora-adapter-improved"

def print_header(text):
    print("\n" + "="*50)
    print(f"🚀 {text.upper()}")
    print("="*50)

print_header("Initializing Improved Side-by-Side Demo")
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print(f"Loading base model to {device.upper()}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32, 
).to(device)

try:
    print(f"Loading (but NOT applying yet) IMPROVED adapters from '{ADAPTER_DIR}'...")
    model = PeftModel.from_pretrained(model, ADAPTER_DIR, adapter_name="professional_improved")
except Exception as e:
    print(f"\n❌ ERROR: Could not find the IMPROVED adapter folder '{ADAPTER_DIR}'.")
    print("Please run 'python3 interactive_train_improved.py' first!")
    sys.exit()

model.eval()

print_header("Improved Model Ready!")
print("Type a 'blunt' email and notice the difference in quality.")
print("Type 'quit' or 'exit' to stop.")

while True:
    user_input = input("\n[BLUNT EMAIL]: ").strip()
    
    if user_input.lower() in ["quit", "exit"]:
        print("Goodbye!")
        break
    
    if not user_input:
        continue

    prompt = f"Rewrite professionally: {user_input}"
    messages = [{"role": "user", "content": prompt}]
    
    inputs = tokenizer.apply_chat_template(
        messages, 
        tokenize=True, 
        add_generation_prompt=True, 
        return_tensors="pt",
        return_dict=True
    ).to(device)

    # Generate - BASE MODEL
    print("\nGenerating Base Response...")
    with torch.no_grad():
        with model.disable_adapter():
            out_base = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
            )

    input_length = inputs["input_ids"].shape[1]
    base_text = tokenizer.decode(out_base[0][input_length:], skip_special_tokens=True).strip()

    # Generate - IMPROVED LORA MODEL
    print("Generating Improved LoRA Response...")
    model.set_adapter("professional_improved")
    with torch.no_grad():
        out_lora = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=True,          # IMPROVED: Using sampling
            temperature=0.7,         # IMPROVED: Added creativity
            repetition_penalty=1.1   # IMPROVED: Preventing loops
        )

    lora_text = tokenizer.decode(out_lora[0][input_length:], skip_special_tokens=True).strip()

    print("\n" + "*" * 50)
    print(f"[CASUAL DEMO -> BASE ONLY]:\n{base_text if base_text else '(Empty output)'}")
    print("-" * 50)
    print(f"[PROFESSIONAL DEMO -> IMPROVED LORA]:\n{lora_text if lora_text else '(Empty output)'}")
    print("*" * 50)
