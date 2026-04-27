# 📧 Gemma 3 Multi-Task Email Assistant: Fine-Tuning Demo

This repository demonstrates how to fine-tune the **Gemma 3 270M** model using **LoRA (Low-Rank Adaptation)** for multiple specialized workplace communication tasks. Instead of one general model, we create specialized "adapters" that can transform blunt emails into professional ones, fix grammar, or adjust the length of the message.

---

## 🚀 Key Features

- **Multi-Adapter Support**: Load multiple LoRA adapters (Tone, Grammar, Length, Instructions) into a single base model.
- **Interactive Training**: Conceptual walkthroughs explaining LoRA, Tokenization, and Hardware.
- **Real-Time Testing**: A side-by-side comparison of the Base model vs. the specialized Fine-tuned adapters.
- **Hardware Optimized**: Automatic detection for **Apple Silicon (MPS)**, **NVIDIA (CUDA)**, and **CPU**.

---

## 📖 Table of Contents
1. [Prerequisites & Setup](#-prerequisites--setup)
2. [Demo Flow: Step-by-Step](#-demo-flow-step-by-step)
3. [Specialized Tasks](#-specialized-tasks)
4. [Conceptual Lessons](#-conceptual-lessons)
5. [Troubleshooting](#-troubleshooting)

---

## 🛠 Prerequisites & Setup

### 1. Hardware
- **Recommended**: Mac (M1/M2/M3) or NVIDIA GPU (8GB+ VRAM).
- **Minimum**: Modern CPU (Note: Training will be slow).

### 2. Installation
```bash
# Clone the repository
git clone <repository-url>
cd finetuning_demo

# Install dependencies
pip install -r requirements.txt
```

### 3. Hugging Face Access
Gemma 3 is a gated model.
1. Accept the license on [Hugging Face](https://huggingface.co/google/gemma-3-270m-it).
2. Create a [Write Token](https://huggingface.co/settings/tokens).
3. Log in:
   ```bash
   huggingface-cli login
   hf auth login
   ```
### 4. RunPod Setup (GPU Environment)

This project was developed and tested using **RunPod** for GPU-based training and inference.

#### Step 1: Create a Pod
- Go to https://runpod.io
- Select a GPU (recommended: **A100 / RTX 4090 / 3090**)
- Choose a template with Python or PyTorch pre-installed

#### Step 2: Clone Repository
```bash
git clone https://github.com/sj74p/finetuning_demo.git
cd finetuning_demo
```

**Note:** Large-scale training was performed on RunPod for faster GPU execution. Since RunPod environments are temporary, any local data or trained adapters will be lost when the pod is stopped. Ensure important files are saved externally.

---

## 🏗 Demo Flow: Step-by-Step

### Step 1: Data Generation
Generate synthetic datasets for each task. Each dataset follows an instruction-based format with an input email and expected output.
```bash
python generate_emails_tone_dataset.py
python generate_emails_grammar_dataset.py
python generate_emails_short_expand_dataset.py
python generate_emails_instruction_dataset.py
```

### Step 2: Interactive Fine-Tuning
Train the LoRA adapters. Each script follows a conceptual walkthrough.
```bash
# Train the Tone adapter (Friendly, Assertive, etc.)
python interactive_train.py

# Train other specialized adapters
python interactive_grammar_train.py
python interactive_short_expand_train.py
python interactive_instruction_train.py
```
The normal adapters are trained with 3 epochs.
This version helps establish a baseline for each task-specific adapter.

### Step 3: Multi-Task Interactive Test
Run the interactive demo to test all adapters in one place. choose the task to perform and type an email.
```bash
python interactive_test.py
```
Supported options include:

- Friendly, assertive, apologetic, persuasive, and professional tone rewriting
- Grammar and clarity correction
- Email shortening and expansion
- Custom instruction-based rewriting
---
### Step 4: Train Improved LoRA Adapters
To train improved versions, follow the same steps as Step 2, but use the *_improved.py scripts:
```bash
# Train improved adapters with 20 epochs
python interactive_train_improved.py
python interactive_grammar_train_improved.py
python interactive_short_expand_train_improved.py
python interactive_instruction_train_improved.py
```
These improved adapters are trained with 15 epochs(Normal adapter trained with 3 epochs), enabling better:
- Instruction adherence
- Output consistency
- Tone accuracy
- Length control
### Step 5: Compare Normal vs Improved Adapters

Run the comparison script:
```bash
python compare_all_models.py
```
This compares:

- Normal adapters (3 epochs)
- Improved adapters (15 epochs)

The comparison highlights improvements in:

- Output quality
- Meaning preservation
- Grammar correction
- Task-specific performance
## 🎯 Specialized Tasks

| Task | Adapter Name | Description |
| :--- | :--- | :--- |
| **Tone** | `tone_adapter` | Friendly, Assertive, Apologetic, Persuasive, Professional. |
| **Grammar** | `grammar_adapter` | Fixes typos, clarity, and phrasing while keeping the meaning. |
| **Length** | `length_adapter` | Shortens or Expands the email as needed. |
| **Custom** | `instruction_adapter` | Follows any custom natural language instruction. |

---

## 🎓 Conceptual Lessons

### 1. LoRA (Low-Rank Adaptation)
We don't train all 270 million parameters. Instead, we train a tiny "adapter" (layers added to the model) that contains only ~1-5% of the total weights. This makes fine-tuning fast and memory-efficient.

### 2. Tokenization & Chat Templates
Models don't read words; they read numbers. We use the **Gemma 3 Tokenizer** to convert text. We also use **Chat Templates** to structure the input so the model understands the difference between a "User" and an "Assistant."

### 3. Hyperparameters
- **Learning Rate**: How big of a "step" the model takes when learning. Too high = collapse; too low = never learns.
- **Epochs**: How many times the model reads the entire "manners book."
- **Repetition Penalty**: Prevents the model from getting stuck in loops (e.g., "Thank you thank you thank you...").

---

## 🛠 Troubleshooting

- **Out of Memory (OOM)**: Reduce per_device_train_batch_size to 1.
- **Gated Model Access**: Make sure you accepted the Gemma license and logged in with Hugging Face.
- **Missing Packages**: Run pip install -r requirements.txt.
- **Slow Training**: Check whether the script is running on CPU instead of CUDA or MPS.
- **Noisy Base Model Output**: The base model may generate explanations, subjects, greetings, or extra text. The fine-tuned adapters are designed to produce more direct task-specific outputs.

---

## ⚖ License
Apache 2.0