from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

MODEL_NAME = "mistralai/Mistral-Nemo-Instruct-2407"

# Login to Hugging Face
login("")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrainged(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

# Load full-precision model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_8bit=False,   # full precision
    device_map="auto",
    cache_dir="/tmp",
    trust_remote_code=True
)

# --- Convert model to FP16 ---
#model = model.half()

# Resize token embeddings if needed
model.resize_token_embeddings(len(tokenizer))

# Optional: check dtype of a parameter to confirm FP16
print(next(model.parameters()).dtype)

"""
Increased max len to 1024
For small training sets, train and test are exactly the same.
"""

MAX_LENGTH = 1024

dataset_name = 'datasets/minji-jennie_training-latest.jsonl'

def tokenize_and_mask(examples):
    inputs = []
    for p, r in zip(examples["prompt"], examples["response"]):
        if p is None or r is None:
            continue

        # concatenate prompt + response
        full = p + r

        # tokenize full text and prompt separately
        tok_full = tokenizer(full, truncation=True, max_length=MAX_LENGTH, padding="max_length")
        tok_prompt = tokenizer(p, truncation=True, max_length=MAX_LENGTH)

        # mask out prompt tokens so that only response is trained
        prompt_len = len(tok_prompt["input_ids"])
        labels = [-100] * len(tok_full["input_ids"])
        for i in range(prompt_len, len(tok_full["input_ids"])):
            labels[i] = tok_full["input_ids"][i]

        tok_full["labels"] = labels
        inputs.append(tok_full)

    return {k: [d[k] for d in inputs] for k in inputs[0]}

from datasets import load_dataset, DatasetDict
# --- load jsonl dataset ---
dataset = load_dataset("json", data_files={"train": dataset_name})

# load raw dataset (two lines in jsonl)
split_dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)


#dataset = load_dataset("json", data_files=dataset_name)["train"]

# duplicate for both train and test
dataset_dict = DatasetDict({
    "train": split_dataset["train"],
    "test": split_dataset["test"]
})

# tokenize both train and test
tokenized_small = dataset_dict.map(
    tokenize_and_mask,
    batched=True,
    remove_columns=dataset_dict["train"].column_names
)

# set format for pytorch training
tokenized_small.set_format(type="torch", columns=["input_ids","attention_mask","labels"])

# Peek at 2 tokenized samples
for i in range(2):
    sample = tokenized_small["train"][i]
    print(f"--- Sample {i} ---")
    print("input_ids:     ", sample["input_ids"][:40])  # print first 40 tokens
    print("attention_mask:", sample["attention_mask"][:40])
    print("labels:        ", sample["labels"][:40])


iter = '103-hand-curated.jsonl'
model_name = f"models-sft/model-iter-{iter}"
fp_16_path = f'models-sft/{model_name}-fp16'
print(fp_16_path)

from transformers import TrainingArguments, Trainer, default_data_collator, EarlyStoppingCallback

# --- Training arguments ---

train_args = TrainingArguments(
    output_dir=model_name,
    per_device_train_batch_size=3,
    gradient_accumulation_steps=1,
    num_train_epochs=1,              # allow more epochs
    learning_rate=5e-7,               # safer small LR
    #weight_decay=0.01,
    #warmup_steps=20,                  # shorter warmup
    logging_steps=5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=4,
    fp16=False,
    report_to="none",
    max_grad_norm=1.0,
)

# --- Trainer setup ---
trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=tokenized_small["train"],
    eval_dataset=tokenized_small["test"],
    data_collator=default_data_collator,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]  # stop if no improvement for 2 evals
)

# --- Train ---
trainer.train()

# --- Convert to FP16 and save ---

model_fp16 = model.half()  # convert weights to FP16
model_fp16.save_pretrained(fp_16_path)
tokenizer.save_pretrained(fp_16_path)

print(f"FP16 model saved to {fp_16_path}")

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
print(f"Uploading files from {fp_16_path} to {repo_id}...")

api.upload_folder(
    folder_path=fp_16_path,
    repo_id=repo_id,
    repo_type="model",
    commit_message=f"Upload fine-tuned FP16 model from {fp_16_path}"
)

print("Model successfully pushed to the Hugging Face Hub!")
print(f"View your model here: https://huggingface.co/{repo_id}")

print("Done Lets Test Before and After")

from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import torch

# --- Paths ---
base_model_name = "mistralai/Mistral-Nemo-Instruct-2407"   # e.g. "EleutherAI/pythia-410m" or "gpt2"
ft_model_name = fp_16_path                 # output_dir from your training
json_file = "datasets/hand_curated_13.jsonl"  # your JSONL file

# --- Load tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# --- Load prompts directly from your JSONL ---
prompts = []
with open(json_file, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        prompts.append(data["prompt"])  # include <START> and formatting exactly

def generate_responses(model, prompts, title):
    print(f"\n--- {title} ---")
    for i, prompt in enumerate(prompts):
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
        print(f"\nPrompt {i+1}:\n{text}\n")

print(" --- Load base model and generate BEFORE fine-tuning --- ")
model_base = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float32)
generate_responses(model_base, prompts, "BEFORE FINE-TUNING")

print(" --- Load fine-tuned model and generate AFTER fine-tuning --- ")
model_ft = AutoModelForCausalLM.from_pretrained(ft_model_name, torch_dtype=torch.float32)
generate_responses(model_ft, prompts, "AFTER FINE-TUNING")

print("Extra Install Stuff")

# For Sanity Check
model_fp16 = model.half()  # convert weights to FP16
model_fp16.save_pretrained(fp_16_path)
tokenizer.save_pretrained(fp_16_path)

# Optional: also save FP32 model if you want
model.save_pretrained(model_name)
tokenizer.save_pretrained(model_name)

print(f"FP32 model saved to {model_name}")
print(f"FP16 model saved to {fp_16_path}")


