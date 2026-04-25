import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sys

MODEL_ID = "google/gemma-3-270m-it"

# Use the adapter name from your updated training script
ADAPTER_DIR = "gemma3-270m-email-instruction-lora-adapter_improved"


def print_header(text):
    print("\n" + "=" * 50)
    print(f"🚀 {text.upper()}")
    print("=" * 50)


def get_dtype(device):
    if device == "cuda":
        return torch.bfloat16
    return torch.float32


def get_instruction(choice):
    mapping = {
        "1": "Rewrite this email in a friendly tone.",
        "2": "Rewrite this email in an assertive tone.",
        "3": "Rewrite this email in an apologetic tone.",
        "4": "Rewrite this email in a persuasive tone.",
        "5": "Rewrite this email in a professional tone.",
        "6": "Fix grammar and improve clarity while keeping the meaning the same.",
        "7": "Shorten this email while keeping the main meaning.",
        "8": "Expand this email into a detailed professional version.",
    }
    return mapping.get(choice)


def generate_response(model, tokenizer, messages, device, use_adapter=True):
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    ).to(device)

    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        if use_adapter:
            output = model.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        else:
            with model.disable_adapter():
                output = model.generate(
                    **inputs,
                    max_new_tokens=120,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

    generated_tokens = output[0][input_length:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


print_header("Initializing Interactive Inference")

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device.upper()}")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Loading base model to {device.upper()}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=get_dtype(device),
    attn_implementation="eager",
).to(device)

try:
    print(f"Loading LoRA adapter from '{ADAPTER_DIR}'...")
    model = PeftModel.from_pretrained(
        model,
        ADAPTER_DIR,
        adapter_name="email_assistant"
    )
except Exception as e:
    print(f"\n❌ ERROR: Could not find adapter folder '{ADAPTER_DIR}'.")
    print("Please train the model first or check your adapter folder name.")
    print(f"Details: {e}")
    sys.exit()

model.eval()

print_header("Model Ready")

while True:
    
    choice = input(
"1. Friendly tone\n"
"2. Assertive tone\n"
"3. Apologetic tone\n"
"4. Persuasive tone\n"
"5. Professional tone\n"
"6. Fix grammar and clarity\n"
"7. Shorten email\n"
"8. Expand email\n"
"9. Custom instruction\n"
"Type 'quit' or 'exit' to stop.\n""Choose task: ").strip()

    if choice.lower() in ["quit", "exit"]:
        print("Goodbye!")
        break

    if choice == "9":
        instruction = input("\nEnter custom instruction: ").strip()
        if not instruction:
            print("Instruction cannot be empty.")
            continue
    else:
        instruction = get_instruction(choice)

    if instruction is None:
        print("Invalid choice. Please choose 1 to 9.")
        continue

    user_input = input("\n[EMAIL]: ").strip()

    if user_input.lower() in ["quit", "exit"]:
        print("Goodbye!")
        break

    if not user_input:
        print("Email cannot be empty.")
        continue

    prompt = f"""Instruction: {instruction} of this
Email: {user_input}"""

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    print("\nGenerating Base Model...")
    base_text = generate_response(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        device=device,
        use_adapter=False
    )

    print("Generating LoRA Model...")
    model.set_adapter("email_assistant")

    lora_text = generate_response(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        device=device,
        use_adapter=True
    )

    print("\n" + "*" * 50)
    print(f"[TASK]: {instruction}")
    print("-" * 50)
    print(f"[INPUT EMAIL]:\n{user_input}")
    print("-" * 50)
    print(f"[BASE MODEL]:\n{base_text if base_text else '(Empty output)'}")
    print("-" * 50)
    print(f"[LORA MODEL]:\n{lora_text if lora_text else '(Empty output)'}")
    print("*" * 50)