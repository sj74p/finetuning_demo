import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sys

MODEL_ID = "google/gemma-3-270m-it"

TONE_ADAPTER_DIR = "gemma3-270m-email-tone-lora-adapter-improved"
GRAMMAR_ADAPTER_DIR = "gemma3-270m-email-grammar-lora-adapter-improved"
LENGTH_ADAPTER_DIR = "gemma3-270m-email-length-lora-adapter-improved"
INSTRUCTION_ADAPTER_DIR = "gemma3-270m-email-instruction-lora-adapter-improved"


def print_header(text):
    print("\n" + "=" * 50)
    print(f"🚀 {text.upper()}")
    print("=" * 50)


def get_dtype(device):
    if device == "cuda":
        return torch.bfloat16
    return torch.float32


def get_task(choice):
    mapping = {
        "1": ("tone_adapter", "Rewrite this email in a friendly tone."),
        "2": ("tone_adapter", "Rewrite this email in an assertive tone."),
        "3": ("tone_adapter", "Rewrite this email in an apologetic tone."),
        "4": ("tone_adapter", "Rewrite this email in a persuasive tone."),
        "5": ("tone_adapter", "Rewrite this email in a professional tone."),
        "6": ("grammar_adapter", "Fix grammar and improve clarity while keeping the meaning exactly the same."),
        "7": ("length_adapter", "Shorten this email while keeping the main meaning."),
        "8": ("length_adapter", "Expand this email into a detailed version."),
        "9": ("instruction_adapter", None),
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
                max_new_tokens=100,
                do_sample=False,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        else:
            with model.disable_adapter():
                output = model.generate(
                    **inputs,
                    max_new_tokens=80,
                    do_sample=False,
                    repetition_penalty=1.2,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

    generated_tokens = output[0][input_length:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


print_header("Initializing Email Assistant Inference")

device = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device.upper()}")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Loading base model to {device.upper()}...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=get_dtype(device),
    attn_implementation="eager",
).to(device)

try:
    print(f"Loading tone LoRA adapter from '{TONE_ADAPTER_DIR}'...")
    model = PeftModel.from_pretrained(
        base_model,
        TONE_ADAPTER_DIR,
        adapter_name="tone_adapter"
    )

    print(f"Loading grammar LoRA adapter from '{GRAMMAR_ADAPTER_DIR}'...")
    model.load_adapter(
        GRAMMAR_ADAPTER_DIR,
        adapter_name="grammar_adapter"
    )

    print(f"Loading length LoRA adapter from '{LENGTH_ADAPTER_DIR}'...")
    model.load_adapter(
        LENGTH_ADAPTER_DIR,
        adapter_name="length_adapter"
    )

    print(f"Loading instruction LoRA adapter from '{INSTRUCTION_ADAPTER_DIR}'...")
    model.load_adapter(
        INSTRUCTION_ADAPTER_DIR,
        adapter_name="instruction_adapter"
    )

except Exception as e:
    print("\n❌ ERROR: Could not load one or more adapter folders.")
    print("Please make sure all adapters are trained and saved correctly.")
    print(f"Details: {e}")
    sys.exit()

model.eval()

print_header("Email Assistant Ready")

while True:
    choice = input(
        "\nChoose task:\n"
        "1. Friendly tone\n"
        "2. Assertive tone\n"
        "3. Apologetic tone\n"
        "4. Persuasive tone\n"
        "5. Professional tone\n"
        "6. Fix grammar and clarity\n"
        "7. Shorten email\n"
        "8. Expand email\n"
        "9. Custom instruction adapter\n"
        "Type 'quit' or 'exit' to stop.\n"
        "Choose task: "
    ).strip()

    if choice.lower() in ["quit", "exit"]:
        print("Goodbye!")
        break

    task_info = get_task(choice)

    if task_info is None:
        print("Invalid choice. Please choose 1 to 9.")
        continue

    adapter_name, instruction = task_info

    if adapter_name == "instruction_adapter":
        instruction = input("\n[INSTRUCTION]: ").strip()

        if instruction.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break

        if not instruction:
            print("Instruction cannot be empty.")
            continue

    user_input = input("\n[EMAIL]: ").strip()

    if user_input.lower() in ["quit", "exit"]:
        print("Goodbye!")
        break

    if not user_input:
        print("Email cannot be empty.")
        continue

    if adapter_name == "grammar_adapter":
        label = "Corrected Email"

    elif adapter_name == "length_adapter" and instruction.startswith("Shorten"):
        label = "Shortened Email"

    elif adapter_name == "length_adapter" and instruction.startswith("Expand"):
        label = "Expanded Email"

    elif adapter_name == "instruction_adapter":
        label = "Output"

    else:
        label = "Rewritten Email"

    prompt = f"""Task: {instruction}

Original Email:
{user_input}

{label}:"""

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

    print(f"Generating {adapter_name} Model...")
    model.set_adapter(adapter_name)

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

    if adapter_name == "tone_adapter":
        print(f"[TONE LORA MODEL]:\n{lora_text if lora_text else '(Empty output)'}")
    elif adapter_name == "grammar_adapter":
        print(f"[GRAMMAR LORA MODEL]:\n{lora_text if lora_text else '(Empty output)'}")
    elif adapter_name == "length_adapter":
        print(f"[LENGTH LORA MODEL]:\n{lora_text if lora_text else '(Empty output)'}")
    elif adapter_name == "instruction_adapter":
        print(f"[INSTRUCTION LORA MODEL]:\n{lora_text if lora_text else '(Empty output)'}")

    print("*" * 50)