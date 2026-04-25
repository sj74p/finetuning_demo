# 🎓 Gemma 3 Fine-Tuning Demo: Professional Email Rewriter

This repository contains a complete, step-by-step demonstration for fine-tuning the **Gemma 3 270M** model using **LoRA (Low-Rank Adaptation)**. The goal is to teach the model how to transform "blunt" or "unprofessional" emails into polite, professional workplace communication.

---

## 🚀 Overview

In this demo, you will:
1.  **Generate** a synthetic dataset of blunt vs. professional email pairs.
2.  **Fine-tune** Gemma 3 using an interactive script that explains core LLM concepts.
3.  **Compare** the performance of the Base model, the Instruct model, and the new Fine-tuned model.
4.  **Test** the results in a live, interactive environment.

---

## 🛠️ Prerequisites & Hardware

### Hardware Requirements
- **Mac (Recommended)**: Apple Silicon (M1, M2, or M3) with at least 8GB of Unified Memory.
- **Windows/Linux**: NVIDIA GPU with at least 8GB VRAM (CUDA support) or a modern CPU (performance will be significantly slower on CPU).

### Environment Setup
1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd Finetuning
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *or run this command:*
    ```bash
    python3 -m pip install -r requirements.txt
    ```

3.  **Hugging Face Setup (Crucial)**:
    - Gemma 3 is a **gated model**. You must go to the [Gemma 3 270M Instruct](https://huggingface.co/google/gemma-3-270m-it) page on Hugging Face and accept the license terms.
    - Create a Hugging Face token [here](https://huggingface.co/settings/tokens).
    - Log in via the terminal:
      ```bash
      huggingface-cli login
      ```
      *If you get "command not found", use this Python version instead:*
      ```bash
      python3 -c "from huggingface_hub import login; login()"
      ```

> [!TIP]
> **Mac Users**: If `huggingface-cli` is not found, you can add its folder to your path permanently:
> `echo 'export PATH="$PATH:$(python3 -m site --user-base)/bin"' >> ~/.zshrc && source ~/.zshrc`


---

## 💻 Hardware-Specific Instructions

The codebase is optimized for **Apple Silicon (MPS)**. If you are on Windows or Linux, follow these adjustments:

### For Windows/Linux (NVIDIA CUDA)
If you have an NVIDIA GPU, the scripts are designed to automatically detect `cuda`. Ensure you have the [NVIDIA drivers](https://www.nvidia.com/download/index.aspx) and [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) installed.

### For Windows/Linux (CPU Only)
If you do not have a GPU, the scripts will fall back to `cpu`. 
> [!NOTE]
> Training on a CPU is very slow. It is recommended to reduce the `num_train_epochs` to `1` in `interactive_train.py` if training on a CPU.

---

## ☁️ Running on RunPod.io (Cloud GPU)

If you don't have a high-end Mac or NVIDIA GPU, you can rent one on [RunPod.io](https://www.runpod.io/). 

### Step-by-Step RunPod Setup:
1.  **Deploy a GPU**: Choose an instance like the **NVIDIA A6000**, **A100**, or even a smaller **RTX 3090/4090**.
2.  **Select Template**: Use the standard **PyTorch** template (e.g., `runpod/pytorch`).
3.  **Connect**: Open the **Web Terminal** or connect via **SSH**.
4.  **Environment Setup**:
    ```bash
    git clone <repository-url>
    cd Finetuning
    pip install -r requirements.txt
    ```
5.  **Hugging Face Login**:
    ```bash
    huggingface-cli login
    ```
6.  **Run Demo**: Follow the "Sequential Demo Flow" below. The scripts will automatically detect and use the RunPod GPU (**CUDA**).

---

## 📖 Sequential Demo Flow

Follow these steps in order to show the demo to students:

### Step 1: Generate the Dataset
Run the generation script to create the `emails.jsonl` file. This is the model's "manners book."
```bash
python3 generate_emails_dataset.py
```

### Step 2: Interactive Training (Concept Walkthrough)
This is the core educational script. It explains **Hardware (MPS)**, **Model Weights**, **LoRA Adapters**, and **Tokenization** while training.
```bash
python3 interactive_train.py
```
*Students can press `y` to proceed after reading each conceptual explanation.*

### Step 3: Live Interactive Test
Show the "Wow" factor. This script allows you to type any blunt email and see a side-by-side comparison of the **Base Model** vs. your **Fine-tuned Model**.
```bash
python3 interactive_test.py
```

### Step 4: Compare All Models (The Ultimate Demo)
To see the full journey of the model, run this script to compare: Base, Instruct, the "Poor" LoRA, and the "Improved" LoRA.
```bash
python3 compare_all_models.py
```

---

## 🎓 Lesson: Why Hyperparameters Matter

If you ran the scripts above and got "garbage" output (like repetitive or nonsensical text), you've just seen a **Model Collapse** in action! This happens when training settings aren't balanced. 

### Part A: The "Failure" Case (`interactive_train.py`)
- **High Learning Rate (`2e-4`)**: The model tried to learn too fast and "overfit" to specific tokens.
- **Low Epochs (`3`)**: The model didn't spend enough time reading the "manners book" to understand the tone.
- **Greedy Decoding**: Without a repetition penalty, the model got stuck in infinite loops.

### Part B: The "Improved" Case (`interactive_train_improved.py`)
To see the difference, run the improved version of the demo:

1. **Train the Improved Model**:
   ```bash
   python3 interactive_train_improved.py
   ```
   *Changes: Lower learning rate (`5e-5`), more epochs (`15`), and a smoother learning curve.*

2. **Test the Improved Model**:
   ```bash
   python3 interactive_test_improved.py
   ```
   *Changes: Adds a `repetition_penalty` and `sampling` to ensure the responses are natural and clean.*

---

## ✍️ Dataset Customization (Project Idea)

You can easily customize the model by adding their own examples. 
1. Open `generate_emails_dataset.py`.
2. Add your own pairs to the `data` list:
   ```python
   data = [
       ("Where is my money?", "I was wondering if there was an update on the payment status?"),
       # Add more here...
   ]
   ```
3. Re-run `python3 generate_emails_dataset.py` and re-train.

### 🤖 Generating More Data with AI
You can use a large LLM (like Gemini or ChatGPT) to generate hundreds of new examples. Here is a prompt you can use:

> "Act as a dataset generator. Generate 50 pairs of workplace communication examples. The first part should be a 'blunt', 'rude', or 'unprofessional' short email or message. The second part should be a 'highly professional', 'polite', and 'constructive' version of that same message. Format it as a Python list of tuples like this: `("blunt message", "professional message"),`"

---

## 🧠 Key Concepts Covered

- **LoRA (Low-Rank Adaptation)**: Instead of training all 270 million parameters, we only train a tiny "adapter" (about 15MB) that sits on top of the model.
- **PEFT (Parameter-Efficient Fine-Tuning)**: The library used to manage LoRA adapters.
- **Tokenization**: Breaking down human text into numbers that the model can understand.
- **Chat Templates**: Formatting the conversation so the model knows who is the user and who is the assistant.

---

## 🛠️ Troubleshooting

### 1. Out of Memory (OOM)
If you get a "CUDA Out of Memory" or "MPS Memory Error":
- Close other applications that use the GPU (like browsers or video players).
- In `interactive_train.py`, try reducing `per_device_train_batch_size` to `1`.

### 2. "Gated Model" Error
If you see an error about unauthorized access to the model:
- Ensure you have accepted the terms on Hugging Face.
- Ensure you have run `huggingface-cli login`.

### 3. "Invalid User Token" (401 Error)
If you get an "Unauthorized" or "Invalid Token" error during login:
- **Re-copy the token**: Go to [Hugging Face Settings](https://huggingface.co/settings/tokens) and re-copy your token.
- **Check Permissions**: Ensure the token is either a **"Write"** token or a "Fine-grained" token with repository read/write access.
- **Paste Carefully**: When pasting into the terminal, the characters will be invisible. Just paste once and hit Enter.

### 4. Training is too slow
If training takes more than 5-10 minutes:
- You might be running on **CPU** instead of a GPU. Check the start-up logs of the script to see which device is being used.

### 5. "Command Not Found" (Mac PATH issue)
If `huggingface-cli` is not recognized even after installation:
- **Temporary Fix**: Use the full path: `~/Library/Python/3.9/bin/huggingface-cli login`
- **Permanent Fix**: Add the folder to your PATH:
  `echo 'export PATH="$PATH:$(python3 -m site --user-base)/bin"' >> ~/.zshrc && source ~/.zshrc`

---

## ⚖️ License
Apache 2.0