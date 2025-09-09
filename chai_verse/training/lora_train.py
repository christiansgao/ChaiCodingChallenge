#####################
# Handcurated LoRA  #
#####################

from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_NAME = "mistralai/Mistral-Nemo-Instruct-2407"

# --- Login to Hugging Face Hub ---
login("")

# --- Load tokenizer and add pad token if missing ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

# --- Load base model in FP16 for efficiency ---
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",       # use fp16 if available
    device_map="auto",
    cache_dir="/tmp",
    trust_remote_code=True
)

# --- Prepare for LoRA fine-tuning (keeps base frozen, enables adapters) ---
model = prepare_model_for_kbit_training(model)

# --- Configure LoRA (tuned for small, stylistic dataset) ---
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj","k_proj"],  # less invasive
    bias="none",
    task_type="CAUSAL_LM"
)

# --- Wrap model with LoRA adapters ---
model = get_peft_model(model, lora_config)

# --- Resize embeddings if tokenizer was modified ---
model.resize_token_embeddings(len(tokenizer))

# --- Sanity check ---
print("Model dtype:", next(model.parameters()).dtype)
print("LoRA params trainable:", sum(p.numel() for p in model.parameters() if p.requires_grad))
print("Total params:", sum(p.numel() for p in model.parameters()))

from datasets import load_dataset, DatasetDict
from transformers import DataCollatorForSeq2Seq

MAX_LENGTH = 1024
dataset_name = "datasets/curated-examples-17.jsonl"

def tokenize_and_mask(examples):
    inputs = []
    for p, r in zip(examples["prompt"], examples["response"]):
        if p is None or r is None:
            continue

        # concatenate prompt + response
        full = p + r

        # tokenize full text and prompt separately (no padding here!)
        tok_full = tokenizer(full, truncation=True, max_length=MAX_LENGTH, padding=False)
        tok_prompt = tokenizer(p, truncation=True, max_length=MAX_LENGTH, padding=False)

        # mask out prompt tokens so only the response contributes to loss
        prompt_len = len(tok_prompt["input_ids"])
        labels = [-100] * len(tok_full["input_ids"])
        for i in range(prompt_len, len(tok_full["input_ids"])):
            labels[i] = tok_full["input_ids"][i]

        tok_full["labels"] = labels
        inputs.append(tok_full)

    return {k: [d[k] for d in inputs] for k in inputs[0]}

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

# Dynamic padding handled here
data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True)

# Final format for torch training
tokenized_small.set_format(type="torch", columns=["input_ids","attention_mask","labels"])

# Peek at 2 tokenized samples
for i in range(2):
    sample = tokenized_small["train"][i]
    print(f"--- Sample {i} ---")
    print("input_ids:     ", sample["input_ids"][:40])  # print first 40 tokens
    print("attention_mask:", sample["attention_mask"][:40])
    print("labels:        ", sample["labels"][:40])


iter = '28-freaky-girlfriends-5-curated.jsonl'

model_name = f"modes-lora/model-iter-{iter}"
merged_path = f"modes-lora/model-iter-{iter}-merged"
print(f"Merged FP16 model to {merged_path}")

from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from peft import PeftModel

# --- Training arguments (1 epoch over your dataset) ---
train_args = TrainingArguments(
    output_dir=model_name,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,   # stabilizes updates
    num_train_epochs=3,              # 1 pass max
    learning_rate=2e-6,              # higher LR than full FT (LoRA trains faster)
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.1,
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True,
    report_to="none",
    max_grad_norm=0.8,
)

# --- Dynamic padding collator ---
data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True)

# --- Trainer setup ---
trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=tokenized_small["train"],
    eval_dataset=tokenized_small["test"],
    data_collator=data_collator,
    tokenizer=tokenizer
)

# --- Train ---
trainer.train()

# --- Save just the LoRA adapter ---
model.save_pretrained(model_name)
tokenizer.save_pretrained(model_name)
print(f"LoRA adapter saved to {model_name}")

# --- Reload base model and tokenizer for merging ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="float16",
    device_map="auto"
)

# Match tokenizer embedding size
base_model.resize_token_embeddings(len(tokenizer))

# --- Load LoRA adapter and merge ---
peft_model = PeftModel.from_pretrained(base_model, model_name)
merged_model = peft_model.merge_and_unload()

# --- Save merged FP16 model ---
merged_model.save_pretrained(merged_path, torch_dtype="float16", safe_serialization=True)
tokenizer.save_pretrained(merged_path)

print(f"Merged FP16 model saved to {merged_path}")

print(f"FP32 model saved to {model_name}")
print(f"FP16 model saved to {merged_path}")


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
print(f"Uploading files from {merged_path} to {repo_id}...")

api.upload_folder(
    folder_path=merged_path,
    repo_id=repo_id,
    repo_type="model",
    commit_message=f"Upload fine-tuned FP16 model from {merged_path}"
)

print("Model successfully pushed to the Hugging Face Hub!")
print(f"View your model here: https://huggingface.co/{repo_id}")

print("Done Lets Test Before and After")

from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import torch

# --- Paths ---
base_model_name = "mistralai/Mistral-Nemo-Instruct-2407"   # e.g. "EleutherAI/pythia-410m" or "gpt2"
ft_model_name = merged_path                 # output_dir from your training
json_file = "datasets/minji-jennie_training-eval.jsonl"  # your JSONL file

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

print(" --- Load fine-tuned model and generate AFTER fine-tuning --- ")
model_ft = AutoModelForCausalLM.from_pretrained(ft_model_name, torch_dtype=torch.float32)
generate_responses(model_ft, prompts, "AFTER FINE-TUNING")


print(" --- Load base model and generate BEFORE fine-tuning --- ")
model_base = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float32)
generate_responses(model_base, prompts, "BEFORE FINE-TUNING")


