# 📧 Gemma 3 Multi-Task Email Assistant Fine Tuning Demo

This project demonstrates how to fine-tune the **Gemma 3 270M** model using **LoRA (Low-Rank Adaptation)** for multiple workplace email transformation tasks.

Instead of relying on a single model, this system uses **task-specific adapters** to handle:

- Tone transformation
- Grammar correction
- Length control, including shorten and expand
- Instruction-based rewriting

---

## 🚀 Key Features

- **Multi-Adapter Architecture**: Separate LoRA adapters for each task
- **Baseline vs Improved Comparison**: 3 epochs vs 15 epochs training
- **Interactive Testing CLI**: Real-time email transformation
- **RunPod GPU Support**: Faster cloud-based training for larger datasets
- **Hardware Flexible**: Supports CPU, Apple Silicon MPS, and NVIDIA CUDA

---

## 🛠 Setup

```bash
git clone https://github.com/sj74p/finetuning_demo.git
cd finetuning_demo
pip install -r requirements.txt
```
Logging into Hugging Face

```bash
huggingface-cli login
hf auth login
```
### Hardware-Specific Instructions
The codebase is optimized for Apple Silicon (MPS). If you are on Windows or Linux, follow these adjustments:

#### Windows/Linux (NVIDIA CUDA)
If you have an NVIDIA GPU, the scripts are designed to automatically detect cuda. Ensure you have the NVIDIA drivers and CUDA Toolkit installed.

#### Windows/Linux (CPU Only)
If you do not have a GPU, the scripts will fall back to cpu.


## Demo Flow: Step-by-Step
### Step 1: Data Generation

Generate synthetic datasets for each task:
```bash
python generate_emails_tone_dataset.py
python generate_emails_grammar_dataset.py
python generate_emails_short_expand_dataset.py
python generate_emails_instruction_dataset.py
```
Each dataset contains:
- Instruction
- Input email
- Expected output

### Step 2: Train Normal LoRA Adapters

Train the baseline version of each adapter:
```bash
python interactive_train.py
python interactive_grammar_train.py
python interactive_short_expand_train.py
python interactive_instruction_train.py
```

The normal adapters are trained with 3 epochs and provide the baseline for comparison.

### Step 3: Multi-Task Interactive Test

Run the interactive demo:
```bash
python interactive_test.py
```
You can:
- Enter an email
- Select a task
View the transformed output

### Step 4: Train Improved LoRA Adapters
To train improved versions, follow the same steps as Step 2, but use the improved scripts:
```bash
python interactive_train_improved.py
python interactive_grammar_train_improved.py
python interactive_short_expand_train_improved.py
python interactive_instruction_train_improved.py
```

The improved adapters are trained with 15 epochs to compare whether longer training improves output quality and instruction-following behavior.

### Step 5: Compare Normal vs Improved Adapters

Run the comparison script:
```bash
python compare_all_models.py
```

This compares:
- Base model
- Normal adapters trained with 3 epochs
- Improved adapters trained with 15 epochs

### Key Learnings
- More epochs do not always improve output quality
- Different tasks respond differently to training duration
- Dataset quality is critical for meaning preservation
- Length transformation is the most challenging task
- Separate adapters reduce task confusion compared to one combined adapter

### RunPod Usage Note
For larger training runs, RunPod was used to leverage GPU acceleration and reduce training time.

RunPod provides a temporary Linux environment, which means:
- Any data, trained adapters, or checkpoints stored locally in the pod may be lost once the pod is stopped or terminated
- Important files should be pushed to GitHub or saved externally
- Trained adapters should be downloaded or stored outside the pod if they need to be reused later
### Troubleshooting
- Out of Memory (OOM): Reduce per_device_train_batch_size
- Gated Model Access: Make sure the Gemma model license is accepted on Hugging Face
- Missing Packages: Run pip install -r requirements.txt
- Slow Training: Check whether the script is running on CPU instead of CUDA or MPS
- Verbose Base Model Output: The base model may generate explanations, subjects, greetings, or extra text. The adapters are designed to produce more direct task-specific outputs
Truncated Outputs: Increase max_new_tokens
Meaning Drift: Improve dataset quality and add more meaning-preserving examples

### License
Apache 2.0