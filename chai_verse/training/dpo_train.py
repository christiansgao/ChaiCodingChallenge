
iter = '202-curated-crazy-gf-1'
iter = '26-freaky-girlfriends-3-curated-dpo'
iter = '26-freaky-girlfriends-3-curated-dpo-9epoch'

model_name = f"models-dpo/model-iter-{iter}"
fp_16_path = f"models-dpo/model-iter-{iter}-fp-16"
print(model_name)
print(fp_16_path)

import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer, DPOConfig

# ----------------------------
# ITERATION TAGGING
# ----------------------------
iter = '202-curated-crazy-gf-1'
iter = '26-freaky-girlfriends-3-curated-dpo'
iter = '26-freaky-girlfriends-4-curated-dpo'

model_name = f"models-dpo/model-iter-{iter}"
fp_16_path = f"models-dpo/model-iter-{iter}-fp-16"

print("Model save dir:", model_name)
print("FP16 final save dir:", fp_16_path)

# ----------------------------
# CONFIGURATION
# ----------------------------
USE_LORA = True
BASE_MODEL = "mistralai/Mistral-Nemo-Instruct-2407"
DATA_FILE = "datasets/crazy-gf-dpo.jsonl"  # {"prompt","chosen","rejected"} per line
OUTPUT_DIR = fp_16_path  # <-- use fp_16_path for training outputs

MAX_LENGTH = 1024
MAX_PROMPT_LENGTH = 768
BATCH_SIZE = 1  # keep small for stability with tiny dataset

torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ----------------------------
# DATA
# ----------------------------
raw_dataset = load_dataset("json", data_files=DATA_FILE)["train"]
split_dataset = raw_dataset.train_test_split(test_size=0.2, seed=42)

# ----------------------------
# TOKENIZER
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ----------------------------
# MODEL (train in FP32)
# ----------------------------
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,  # train in FP32
    device_map="auto"
)

if USE_LORA:
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

# ----------------------------
# DPO CONFIG (train in FP32)
# ----------------------------
dpo_args = DPOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=1,
    num_train_epochs=6,
    learning_rate=1e-7,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.0,
    logging_steps=1,
    evaluation_strategy="steps",
    eval_steps=1,
    save_strategy="no",
    fp16=False,  # no fp16 training
    bf16=False,
    report_to="none",
    max_grad_norm=1.0,
    remove_unused_columns=False,
)

# ----------------------------
# DPO TRAINER
# ----------------------------
dpo_trainer = DPOTrainer(
    model=model,
    args=dpo_args,
    train_dataset=split_dataset["train"],
    eval_dataset=split_dataset["test"],
    tokenizer=tokenizer,
    beta=0.1,
    max_length=MAX_LENGTH,
    max_target_length=MAX_LENGTH,
    max_prompt_length=MAX_PROMPT_LENGTH,
)

# ----------------------------
# TRAIN (FP32)
# ----------------------------
dpo_trainer.train()

# ----------------------------
# SAVE FINAL MODEL (convert to FP16)
# ----------------------------
os.makedirs(fp_16_path, exist_ok=True)
model.half().save_pretrained(fp_16_path)
tokenizer.save_pretrained(fp_16_path)

print(f"✅ DPO training complete. Model trained in FP32 but saved in FP16 at {fp_16_path}")

# ----------------------------
# OPTIONAL: SAVE MERGED MODEL
# ----------------------------
if USE_LORA:
    # merge LoRA adapters into the base model
    merged_model = model.merge_and_unload()
    fp_16_path_merged = fp_16_path + "_merged"
    os.makedirs(fp_16_path_merged, exist_ok=True)
    merged_model.half().save_pretrained(fp_16_path_merged)
    tokenizer.save_pretrained(fp_16_path_merged)

    print(f"✅ Merged FP16 model saved at {fp_16_path_merged}")
else:
    merged_model = model
    fp_16_path_merged = fp_16_path

from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import HfApi
import os

# --- Set names and paths ---
model_name = "fine-tuned-bluemoon"        # your internal name
repo_name = "test-bluemoon-model"         # Hugging Face repo name
username = "christiansgao"                # your HF username

# --- Login via huggingface-cli beforehand or in code ---
# from huggingface_hub import login
# login("YOUR_HF_TOKEN")

# --- Initialize HF API ---
api = HfApi()

# Create repo if it doesn't exist
'''
api.create_repo(
    repo_id=f"{username}/{repo_name}",
    token=True,          # uses the logged-in token
    exist_ok=True        # don't fail if repo already exists
)'''

# --- Upload the model folder to the HF Hub ---
repo_id = f"{username}/{repo_name}"
print(f"Uploading files from {fp_16_path_merged} to {repo_id}...")

api.upload_folder(
    folder_path=fp_16_path_merged,
    repo_id=repo_id,
    repo_type="model",
    commit_message=f"Upload fine-tuned FP16 model from {fp_16_path}"
)

print("Model successfully pushed to the Hugging Face Hub!")
print(f"View your model here: https://huggingface.co/{repo_id}")

print("Done Lets Test Before and After")

import json
import random
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Paths ---
base_model_name = BASE_MODEL
ft_model_name = fp_16_path_merged
json_file = "datasets/crazy-gf-curated.jsonl"

# --- Settings ---
num_previews = 3   # number of previews to show
num_eval = 20      # number of random samples to evaluate for avg/median
random.seed(42)    # fixed seed for reproducibility

# --- Load tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# --- Load prompts ---
prompts = []
with open(json_file, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        prompts.append(data["prompt"])

# select prompts for preview and evaluation
preview_prompts = random.sample(prompts, min(num_previews, len(prompts)))
eval_prompts = random.sample(prompts, min(num_eval, len(prompts)))

def generate_and_measure(model, all_prompts, preview_prompts, title):
    print(f"\n--- {title} ---")
    word_counts = []
    previews = []

    for i, prompt in enumerate(all_prompts):
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        continuation = text[len(prompt):].strip()
        wc = len(continuation.split())
        word_counts.append(wc)

        if prompt in preview_prompts:
            previews.append((prompt, continuation, wc))

    # print previews
    for j, (prompt, continuation, wc) in enumerate(previews, 1):
        print(f"\nPrompt {j}:\n{prompt}\n--- Response ---\n{continuation}\n[Word count: {wc}]")

    avg_wc = np.mean(word_counts)
    median_wc = np.median(word_counts)
    return avg_wc, median_wc

print(" --- Load fine-tuned model and generate AFTER fine-tuning --- ")
model_ft = AutoModelForCausalLM.from_pretrained(ft_model_name, torch_dtype=torch.float32)
avg_ft, med_ft = generate_and_measure(model_ft, eval_prompts, preview_prompts, "AFTER FINE-TUNING")

print("\n --- Load base model and generate BEFORE fine-tuning --- ")
model_base = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float32)
avg_base, med_base = generate_and_measure(model_base, eval_prompts, preview_prompts, "BEFORE FINE-TUNING")

print("\n=== Word Count Comparison ===")
print(f"Before Fine-Tuning:  Avg = {avg_base:.2f}, Median = {med_base:.2f}")
print(f"After  Fine-Tuning:  Avg = {avg_ft:.2f}, Median = {med_ft:.2f}")


