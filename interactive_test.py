import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sys

MODEL_ID = "google/gemma-3-270m-it"
ADAPTER_DIR = "gemma3-270m-email-lora-adapter"

def print_header(text):
    print("\n" + "="*50)
    print(f"🚀 {text.upper()}")
    print("="*50)

print_header("Initializing Interactive Inference")
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print(f"Loading base model to {device.upper()}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32, 
).to(device)

try:
    print(f"Loading (but NOT applying yet) your fine-tuned LoRA adapters from '{ADAPTER_DIR}'...")
    # Using adapter_name allows us to toggle it on and off dynamically
    model = PeftModel.from_pretrained(model, ADAPTER_DIR, adapter_name="professional")
except Exception as e:
    print(f"\n❌ ERROR: Could not find the trained adapter folder '{ADAPTER_DIR}'.")
    print("Please run 'python3 interactive_train.py' first to train the model!")
    sys.exit()

model.eval()

print_header("Model Ready!")
print("Type a 'blunt' email and the model will rewrite it professionally.")
print("Type 'quit' or 'exit' to stop.")

while True:
    user_input = input("\n[BLUNT EMAIL]: ").strip()
    
    if user_input.lower() in ["quit", "exit"]:
        print("Goodbye!")
        break
    
    if not user_input:
        continue

    # Format the prompt using the Gemma Chat Template
    prompt = f"Rewrite professionally: {user_input}"
    messages = [{"role": "user", "content": prompt}]
    
    # Pre-process
    inputs = tokenizer.apply_chat_template(
        messages, 
        tokenize=True, 
        add_generation_prompt=True, 
        return_tensors="pt",
        return_dict=True
    ).to(device)

    # Generate - BASE MODEL (Casual)
    print("\nGenerating Base (Casual) Response...")
    with torch.no_grad():
        with model.disable_adapter():
            out_base = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
            )

    # Decode - BASE MODEL
    input_length = inputs["input_ids"].shape[1]
    base_tokens = out_base[0][input_length:]
    base_text = tokenizer.decode(base_tokens, skip_special_tokens=True).strip()

    # Generate - LORA MODEL (Professional)
    print("Generating LoRA (Professional) Response...")
    model.set_adapter("professional")
    with torch.no_grad():
        out_lora = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
        )

    # Decode - LORA MODEL
    lora_tokens = out_lora[0][input_length:]
    lora_text = tokenizer.decode(lora_tokens, skip_special_tokens=True).strip()

    print("\n" + "*" * 50)
    print(f"[CASUAL DEMO -> BASE ONLY]:\n{base_text if base_text else '(Empty output)'}")
    print("-" * 50)
    print(f"[PROFESSIONAL DEMO -> BASE + LORA]:\n{lora_text if lora_text else '(Empty output)'}")
    print("*" * 50)
