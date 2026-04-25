import torch
import sys
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model


def ask_to_proceed(step_name):
    print(f"\n[STEP] {step_name}")
    choice = input("Shall we proceed? (y/n): ").lower()
    if choice != "y":
        print("Exiting...")
        sys.exit()


def print_explanation(title, text):
    print("\n" + "=" * 50)
    print(f"🔹 {title.upper()}")
    print("=" * 50)
    print(text)
    print("=" * 50)


# 1) Hardware
print_explanation(
    "Step 1: Hardware Setup",
    "Checking whether we can use MPS, CUDA, or CPU for training."
)

ask_to_proceed("Initialize Hardware")
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device.upper()}")


# 2) Load Model
print_explanation(
    "Step 2: Load Gemma Model",
    "Loading Gemma 3 270M instruction-tuned model and tokenizer."
)

ask_to_proceed("Load Model and Tokenizer")

MODEL_ID = "google/gemma-3-270m-it"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    attn_implementation="eager",
).to(device)

print("✅ Model loaded successfully.")


# 3) Baseline Test
print_explanation(
    "Step 3: Baseline Test",
    "Testing the base model before fine-tuning."
)

test_instruction = "Rewrite this email in a friendly tone."
test_email = "Send me the report ASAP."

test_prompt_text = f"""Instruction: {test_instruction}
Email: {test_email}"""

ask_to_proceed("Test Base Model")

print(f"\n[PROMPT]\n{test_prompt_text}")

messages = [{"role": "user", "content": test_prompt_text}]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True
).to(device)

print("Generating base model response...")

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

input_length = inputs["input_ids"].shape[1]
generated_tokens = out[0][input_length:]
base_output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

print(f"\n[BASE MODEL RESPONSE]\n{base_output_text}\n")


# 4) Apply LoRA
print_explanation(
    "Step 4: Apply LoRA",
    "Applying LoRA adapters so we train only a small number of parameters."
)

ask_to_proceed("Apply LoRA Adapters")

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# 5) Load Dataset
print_explanation(
    "Step 5: Load Instruction Dataset",
    "Loading emails_tone_based.jsonl with instruction, input, and output fields."
)

ask_to_proceed("Load and Format Dataset")

ds = load_dataset("json", data_files="emails_tone_based.jsonl", split="train")


def format_prompts(example):
    user_prompt = f"""Instruction: {example["instruction"]}
Email: {example["input"]}"""

    messages = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": example["output"]}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    return {"text": text}


ds = ds.map(format_prompts)

print("\n[FORMATTED TRAINING SAMPLE]")
print(ds[0]["text"][:500])

ds = ds.train_test_split(test_size=0.1, seed=42)
train_ds = ds["train"]
eval_ds = ds["test"]

print(f"\n📊 Training examples: {len(train_ds)}")
print(f"📊 Evaluation examples: {len(eval_ds)}")


# 6) Training Config
print_explanation(
    "Step 6: Train Model",
    "Fine-tuning the model to follow email transformation instructions."
)

ask_to_proceed("Start Training")

args = SFTConfig(
    output_dir="gemma3-270m-email-tone-lora-improved",
    use_cpu=(device == "cpu"),
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    num_train_epochs=15,
    logging_steps=5,
    eval_strategy="steps",
    eval_steps=10,
    save_steps=10,
    save_total_limit=2,
    report_to="none",
    max_length=512,
    dataset_text_field="text",
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    args=args,
)

trainer.train()


# 7) Save Adapter
print_explanation(
    "Step 7: Save Adapter",
    "Saving the trained LoRA adapter for the instruction-based email assistant."
)

ask_to_proceed("Save Adapter")

ADAPTER_PATH = "gemma3-270m-email-tone-lora-adapter-improved"

trainer.model.save_pretrained(ADAPTER_PATH)
tokenizer.save_pretrained(ADAPTER_PATH)

print(f"\n✅ Adapter saved to: {ADAPTER_PATH}")


# 8) Final Test
print_explanation(
    "Step 8: Final Test",
    "Testing the fine-tuned model using the same prompt."
)

ask_to_proceed("Test Fine-Tuned Model")

model.eval()
model.config.use_cache = True

print(f"\n[PROMPT]\n{test_prompt_text}")
print("Generating fine-tuned response...")

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

generated_tokens = out[0][input_length:]
lora_output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

print("\n" + "*" * 50)
print(f"BEFORE BASE MODEL:\n{base_output_text}")
print("*" * 50)
print(f"AFTER FINE-TUNED MODEL:\n{lora_output_text}")
print("*" * 50)

print("\nNext, run: python interactive_test.py")