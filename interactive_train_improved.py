import torch
import sys
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

def ask_to_proceed(step_name):
    print(f"\n[STEP] {step_name}")
    choice = input("Shall we proceed? (y/n): ").strip().lower()
    if choice not in ['y', 'yes', '']:
        print("Exiting...")
        sys.exit()

def print_explanation(title, text):
    print("\n" + "="*50)
    print(f"🔹 {title.upper()}")
    print("="*50)
    print(text)
    print("="*50)

# 1) Setup
print_explanation("Step 1: The Engine (Hardware)", 
    "🚀 WE ARE CHECKING YOUR HARDWARE POWERHOUSE.\n"
    "We use 'MPS' on Mac or 'CUDA' on Windows/Linux to let your GPU handle the heavy lifting.")

ask_to_proceed("Initialize Hardware")
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Hardware Ready: Using {device.upper()} acceleration.")

# 2) Load Model
print_explanation("Step 2: The Brain (The Model)", 
    "🧠 LOADING THE 'GEMMA 3 270M' BRAIN.")

ask_to_proceed("Load Model & Tokenizer")
MODEL_ID = "google/gemma-3-270m-it"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    attn_implementation="eager",
).to(device)
print("✅ Model loaded!")

# --- BASE MODEL EXAMPLE ---
test_prompt_text = "Rewrite professionally: This code is garbage and broke the build."
messages = [{"role": "user", "content": test_prompt_text}]
inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(device)

print("\n[PROMPT]:", test_prompt_text)
print("Generating baseline...")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
input_length = inputs["input_ids"].shape[1]
base_output_text = tokenizer.decode(out[0][input_length:], skip_special_tokens=True).strip()
print(f"❌ BEFORE (Base Model):\n{base_output_text}\n")

# 3) LoRA Setup
print_explanation("Step 3: The Improved Post-it Note (LoRA)", 
    "📝 APPLYING LoRA CONFIGURATION.\n"
    "We keep the same rank (r=16) but we will be much more careful with how we train it.")

ask_to_proceed("Apply LoRA Adapters")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 4) Load Dataset
ask_to_proceed("Load and Format Data")
ds = load_dataset("json", data_files="emails.jsonl", split="train")

def format_prompts(ex):
    messages = [
        {"role": "user", "content": ex["instruction"]},
        {"role": "assistant", "content": ex["output"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}

ds = ds.map(format_prompts)
ds = ds.train_test_split(test_size=0.1, seed=42)
train_ds, eval_ds = ds["train"], ds["test"]

# 5) Improved Trainer Config
print_explanation("Step 5: The 'Masterpiece' Session (Improved Training)", 
    "⏳ STARTING THE IMPROVED TRAINING.\n"
    "Key Changes for Students:\n"
    "1. Lower Learning Rate (5e-5): Small, careful steps to avoid 'blowing up' the model.\n"
    "2. More Epochs (15): We read the manners book 15 times instead of 3.\n"
    "3. Cosine Scheduler: The learning rate starts high and gently tapers off.\n"
    "4. Weight Decay: Prevents the model from memorizing examples too hard.")

ask_to_proceed("Start Improved Training Loop")
args = SFTConfig(
    output_dir="gemma3-270m-email-lora-improved",
    use_cpu=(device == "cpu"),
    per_device_train_batch_size=1, # Reduced from 4
    gradient_accumulation_steps=8, # Increased from 2 to maintain total batch=8
    dataloader_pin_memory=False,   # CRITICAL: Fix for MPS segmentation fault
    learning_rate=5e-5, 
    num_train_epochs=15, 
    weight_decay=0.01,   
    lr_scheduler_type="cosine", 
    logging_steps=1,
    eval_strategy="steps",
    eval_steps=10,
    save_steps=50,
    save_total_limit=1,
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

# 6) Save
ADAPTER_PATH = "gemma3-270m-email-lora-adapter-improved"
ask_to_proceed(f"Save the improved adapter to {ADAPTER_PATH}")
trainer.model.save_pretrained(ADAPTER_PATH)
tokenizer.save_pretrained(ADAPTER_PATH)

# 7) The Final Exam
print_explanation("Step 7: The Improved Final Exam", 
    "We add 'Repetition Penalty' and 'Sampling' to make the response even more natural.")

ask_to_proceed("Test the Improved Model")
model.eval()
with torch.no_grad():
    out = model.generate(
        **inputs, 
        max_new_tokens=64, 
        do_sample=True,          # Enabled sampling
        temperature=0.7,         # Added for creativity
        repetition_penalty=1.1   # CRITICAL: Prevents 'the most the most' loops
    )
    
lora_output_text = tokenizer.decode(out[0][input_length:], skip_special_tokens=True).strip()

print("\n" + "*"*40)
print(f"❌ BEFORE (Base Model):\n{base_output_text}")
print("*"*40)
print(f"✅ AFTER (Improved LoRA):\n{lora_output_text}")
print("*"*40 + "\n")

print("👉 To run the live side-by-side improved demo: python3 interactive_test_improved.py")
