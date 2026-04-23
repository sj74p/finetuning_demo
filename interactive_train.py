import torch
import sys
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

def ask_to_proceed(step_name):
    print(f"\n[STEP] {step_name}")
    choice = input("Shall we proceed? (y/n): ").lower()
    if choice != 'y':
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
    "Visualization: Think of the CPU as a 'math teacher' and the GPU as a 'room full of calculators'.\n"
    "We use 'MPS' on Mac or 'CUDA' on Windows/Linux to let your GPU handle the heavy lifting.\n"
    "Without this, training would feel like walking through mud.")

ask_to_proceed("Initialize Hardware")
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Hardware Ready: Using {device.upper()} acceleration.")

# 2) Load Model
print_explanation("Step 2: The Brain (The Model)", 
    "🧠 LOADING THE 'GEMMA 3 270M' BRAIN.\n"
    "Visualization: 270 Million Parameters means the model has 270,000,000 'tuning knobs'.\n"
    "Here is what a tiny piece of that 'brain' looks like (weights):\n"
    "   [[ 0.012, -0.045,  0.089 ],\n"
    "    [-0.033,  0.112, -0.007 ],\n"
    "    [ 0.056, -0.021,  0.044 ]]\n"
    "We load it in 'float32' mode to keep these tiny decimals precise.")

ask_to_proceed("Load Model & Tokenizer")
MODEL_ID = "google/gemma-3-270m-it"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    attn_implementation="eager",
).to(device)
print("✅ Model loaded! It's currently a general-purpose 'it' model.")

# --- NEW: BASE MODEL EXAMPLE ---
print_explanation("Step 2.5: The Baseline (Before Training)", 
    "Let's see how the model behaves BEFORE we teach it any manners.\n"
    "We will ask it to rewrite a blunt email, and we'll compare it at the end.")

test_prompt_text = "Rewrite professionally: This code is garbage and broke the build."

ask_to_proceed("Test the Base Model")
print(f"\n[PROMPT]: {test_prompt_text}")
messages = [{"role": "user", "content": test_prompt_text}]
inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(device)

print("Generating...")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
input_length = inputs["input_ids"].shape[1]
generated_tokens = out[0][input_length:]
base_output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

print(f"\n[BASE MODEL RESPONSE]:\n{base_output_text}\n")
print("Notice how it might just repeat the prompt, answer poorly, or lack the specific professional tone we want.")
# -------------------------------

# 3) LoRA Setup
print_explanation("Step 3: The Post-it Note (LoRA)", 
    "📝 APPLYING LoRA (Low-Rank Adaptation).\n"
    "Visualization: Instead of changing the massive 270M parameters (W),\n"
    "we freeze W and add a small 'adapter': W' = W + (A * B).\n"
    "Matrix A reduces the dimension from d to a tiny 'rank' (r=16),\n"
    "and Matrix B expands it back. This creates a tiny bottleneck that forces\n"
    "the model to only learn the MOST important 'tone' features.\n\n"
    "Parameters: 270M (Base) vs ~3.7M (LoRA). We only train the 3.7M!")

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
model.print_trainable_parameters()

# 4) Load Dataset
print_explanation("Step 4: The Schooling (Dataset)", 
    "📚 PREPARING YOUR EMAIL DATASET.\n"
    "Visualization: The model doesn't read letters, it reads NUMBERS (Token IDs).\n"
    "Example Translation:\n"
    "   Text:  'Please send...' \n"
    "   Tokens: [ 2341, 1402 ]\n"
    "We wrap these in a conversation template so the model knows who is talking.")

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

# Show a sample of tokenization and formatting
sample_raw = "Professional"
sample_text = ds[0]['text']
sample_tokens = tokenizer.encode(sample_text)[:10]

print("\n[PROMPT FORMATTING EXAMPLE]")
print("This is what format_prompt() does. It wraps the text in Gemma's specific 'Chat Template':")
print(f"Raw String:\n{repr(sample_text[:120])}...\n")
print("Notice the <start_of_turn> and <end_of_turn> tokens? These are 'control tokens'\n"
      "that tell the model exactly when the human stops talking and the AI should start.")

print(f"\n[TOKENIZATION EXAMPLE]")
print("The model doesn't read letters, and it often breaks words into chunks (Byte-Pair Encoding).")
print(f"Example Word: '{sample_raw}' -> Token ID: {tokenizer.encode(sample_raw, add_special_tokens=False)}")
print("Example Prefix+Word: ' Unprofessional' -> Token IDs:", tokenizer.encode(" Unprofessional", add_special_tokens=False))
print(f"\nFull sentence token IDs (first 10): {sample_tokens} ...")

ds = ds.train_test_split(test_size=0.1, seed=42)
train_ds, eval_ds = ds["train"], ds["test"]
print(f"\n📊 Training on {len(train_ds)} manners examples. Testing on {len(eval_ds)} hidden examples.")

# 5) Trainer Config
print_explanation("Step 5: The Study Session (Training)", 
    "⏳ STARTING THE FINE-TUNING LOOP.\n"
    "Visualization: Think of this as an 'exam'. The model tries to guess the answer,\n"
    "looks at the 'Loss' (the error score), and adjusts its Post-it Note to do better next time.\n"
    "We will run 3 Epochs—meaning the model reads the entire manners book 3 times.")

ask_to_proceed("Start Training Loop")
args = SFTConfig(
    output_dir="gemma3-270m-interactive-lora",
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

# 6) Save
print_explanation("Step 6: Graduating (Saving)", 
    "🎓 SAVING YOUR TRAINED ADAPTER.\n"
    "Visualization: We don't save the whole library. We only save the Post-it Note!\n"
    "The file is only ~15MB. You can share this tiny file with anyone who has Gemma 3,\n"
    "and they will instantly have your 'Professional Email' version.")

ask_to_proceed("Save the manners-adapter")
ADAPTER_PATH = "gemma3-270m-email-lora-adapter"
trainer.model.save_pretrained(ADAPTER_PATH)
tokenizer.save_pretrained(ADAPTER_PATH)
print(f"\n🎉 GRADUATION COMPLETE! Adapter saved to {ADAPTER_PATH}")

# --- NEW: LORA MODEL EXAMPLE ---
print_explanation("Step 7: The Final Exam", 
    "Let's test our trained 'Professional' adapter using the exact same prompt\n"
    "we gave the Base model in Step 2.5.")

ask_to_proceed("Test the Trained Model")
print(f"\n[PROMPT]: {test_prompt_text}")
print("Generating...")

model.config.use_cache = True
# Put model in eval mode to disable dropout before generating
model.eval()

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    
generated_tokens = out[0][input_length:]
lora_output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

print("\n" + "*"*40)
print(f"❌ BEFORE (Base Model):\n{base_output_text}")
print("*"*40)
print(f"✅ AFTER (LoRA Adapter):\n{lora_output_text}")
print("*"*40 + "\n")

print("👉 To see the modular demo in action, run: python3 interactive_test.py")
