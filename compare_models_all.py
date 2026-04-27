import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sys

MODEL_ID = "google/gemma-3-270m-it"

# Normal adapters
TONE_NORMAL = "gemma3-270m-email-tone-lora-adapter"
GRAMMAR_NORMAL = "gemma3-270m-email-grammar-lora-adapter"
LENGTH_NORMAL = "gemma3-270m-email-length-lora-adapter"
INSTRUCTION_NORMAL = "gemma3-270m-email-instruction-lora-adapter"

# Improved adapters
TONE_IMPROVED = "gemma3-270m-email-tone-lora-adapter-improved"
GRAMMAR_IMPROVED = "gemma3-270m-email-grammar-lora-adapter-improved"
LENGTH_IMPROVED = "gemma3-270m-email-length-lora-adapter-improved"
INSTRUCTION_IMPROVED = "gemma3-270m-email-instruction-lora-adapter-improved"

device = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)


def get_dtype(device):
    return torch.bfloat16 if device == "cuda" else torch.float32


def build_prompt(task, email, label):
    return f"""Task: {task}

Original Email:
{email}

{label}:"""


def generate_response(model, tokenizer, prompt, adapter_name=None):
    messages = [{"role": "user", "content": prompt}]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    ).to(device)

    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        if adapter_name:
            model.set_adapter(adapter_name)
            output = model.generate(
                **inputs,
                max_new_tokens=80,
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


print(f"🚀 BASE VS NORMAL VS IMPROVED COMPARISON | Device: {device.upper()}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=get_dtype(device),
    attn_implementation="eager",
).to(device)

try:
    print("Loading normal and improved adapters...")

    model = PeftModel.from_pretrained(
        base_model,
        TONE_NORMAL,
        adapter_name="tone_normal"
    )

    model.load_adapter(TONE_IMPROVED, adapter_name="tone_improved")

    model.load_adapter(GRAMMAR_NORMAL, adapter_name="grammar_normal")
    model.load_adapter(GRAMMAR_IMPROVED, adapter_name="grammar_improved")

    model.load_adapter(LENGTH_NORMAL, adapter_name="length_normal")
    model.load_adapter(LENGTH_IMPROVED, adapter_name="length_improved")

    model.load_adapter(INSTRUCTION_NORMAL, adapter_name="instruction_normal")
    model.load_adapter(INSTRUCTION_IMPROVED, adapter_name="instruction_improved")

except Exception as e:
    print("\n❌ Could not load one or more adapters.")
    print("Please check that all adapter folders exist.")
    print(f"Details: {e}")
    sys.exit()

model.eval()

test_cases = [
    {
        "group": "Tone",
        "normal_adapter": "tone_normal",
        "improved_adapter": "tone_improved",
        "task": "Rewrite this email in a professional tone.",
        "email": "This code is garbage and broke the build.",
        "label": "Rewritten Email"
    },
    {
        "group": "Grammar",
        "normal_adapter": "grammar_normal",
        "improved_adapter": "grammar_improved",
        "task": "Fix grammar and improve clarity while keeping the meaning exactly the same.",
        "email": "i didnt got update from you",
        "label": "Corrected Email"
    },
    {
        "group": "Shorten",
        "normal_adapter": "length_normal",
        "improved_adapter": "length_improved",
        "task": "Shorten this email while keeping the main meaning.",
        "email": "I just wanted to check in again and see if there are any updates from your side regarding this issue.",
        "label": "Shortened Email"
    },
    {
        "group": "Expand",
        "normal_adapter": "length_normal",
        "improved_adapter": "length_improved",
        "task": "Expand this email into a detailed version.",
        "email": "send update",
        "label": "Expanded Email"
    },
    {
        "group": "Instruction",
        "normal_adapter": "instruction_normal",
        "improved_adapter": "instruction_improved",
        "task": "Rewrite this email in a polite and confident tone.",
        "email": "Send me the report ASAP.",
        "label": "Output"
    }
]

for case in test_cases:
    prompt = build_prompt(
        task=case["task"],
        email=case["email"],
        label=case["label"]
    )

    print("Generating base model response...")
    base_output = generate_response(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        adapter_name=None
    )

    print(f"Generating {case['normal_adapter']} response...")
    normal_output = generate_response(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        adapter_name=case["normal_adapter"]
    )

    print(f"Generating {case['improved_adapter']} response...")
    improved_output = generate_response(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        adapter_name=case["improved_adapter"]
    )

    print("\n" + "=" * 80)
    print(f"TASK GROUP: {case['group']}")
    print(f"TASK: {case['task']}")
    print(f"INPUT: {case['email']}")
    print("-" * 80)
    print("[BASE MODEL OUTPUT]")
    print(base_output if base_output else "(Empty output)")
    print("-" * 80)
    print("[NORMAL ADAPTER OUTPUT]")
    print(normal_output if normal_output else "(Empty output)")
    print("-" * 80)
    print("[IMPROVED ADAPTER OUTPUT]")
    print(improved_output if improved_output else "(Empty output)")
    print("=" * 80)

print("\n✅ Normal vs improved comparison complete.")