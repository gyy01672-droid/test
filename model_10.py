import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import torch
import pandas as pd
from datasets import Dataset
from transformers import LlamaTokenizer, LlamaForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import swanlab
import re
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split


def readdata(filename):
    df = pd.read_csv(filename, sep='\t', engine='python')
    df.columns = [c.strip() for c in df.columns]
    df = df[['Text', 'Label']]
    df.columns = ['sentence', 'Label']
    df['sentence'] = df['sentence'].astype(str).str.replace('á', ' ').str.lower().str.strip()
    df['Label'] = df['Label'].astype(str).str.strip()
    return df


df_train_full = readdata('train.csv')
df_test = readdata('test.csv')

unique_labels = sorted(df_train_full['Label'].unique())
label_map = {label: idx for idx, label in enumerate(unique_labels)}
df_train_full['label'] = df_train_full['Label'].map(label_map).astype(int)
df_test['label'] = df_test['Label'].map(label_map).fillna(0).astype(int)

try:
    df_train, df_dev = train_test_split(df_train_full, test_size=0.2, random_state=42, stratify=df_train_full['label'])
except ValueError:
    df_train, df_dev = train_test_split(df_train_full, test_size=0.2, random_state=42)


def build_prompt(sentence, label=None):
    label_desc = ", ".join([f"{v}={k}" for k, v in sorted(label_map.items(), key=lambda x: x[1])])
    if label is not None:
        prompt = f"""### Instruction:
Classify the emotion of the sentence into one of 7 categories: {label_desc}. Respond with only the label number (0-6).

### Input:
{sentence}

### Response:
{label}"""
    else:
        prompt = f"""### Instruction:
Classify the emotion of the sentence into one of 7 categories: {label_desc}. Respond with only the label number (0-6).

### Input:
{sentence}

### Response:
"""
    return prompt


train_texts = [build_prompt(s, l) for s, l in zip(df_train['sentence'], df_train['label'])]
train_dataset = Dataset.from_dict({'text': train_texts})
dev_texts = [build_prompt(s, l) for s, l in zip(df_dev['sentence'], df_dev['label'])]

tokenizer = LlamaTokenizer.from_pretrained(
    '/root/autodl-fs/llama-3.1-8b-instruct',
    use_fast=False,
    local_files_only=True
)
tokenizer.pad_token = tokenizer.eos_token


def tokenize(example):
    texts = example['text']
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=256
    )

    labels = []
    for i, text in enumerate(texts):
        input_ids = tokenized["input_ids"][i]
        label_ids = input_ids.copy()

        response_text = "### Response:\n"
        response_tokens = tokenizer(response_text, add_special_tokens=False)["input_ids"]

        def find_sublist(lst, sub):
            for j in range(len(lst) - len(sub) + 1):
                if lst[j:j + len(sub)] == sub:
                    return j + len(sub)
            return -1

        start = find_sublist(input_ids, response_tokens)

        if start == -1:
            label_ids = [-100] * len(label_ids)
        else:
            label_ids[:start] = [-100] * start

        labels.append(label_ids)

    tokenized["labels"] = labels
    return tokenized


train_dataset = train_dataset.map(tokenize, batched=True)
dev_dataset = Dataset.from_dict({'text': dev_texts})
dev_dataset = dev_dataset.map(tokenize, batched=True)

model = LlamaForCausalLM.from_pretrained(
    '/root/autodl-fs/llama-3.1-8b-instruct',
    local_files_only=True,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map='auto'
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

run = swanlab.init(project="model", experiment_name="model")

training_args = TrainingArguments(
    output_dir="./llama-lora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    learning_rate=5e-5,
    save_steps=100,
    fp16=True,
    bf16=False,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    save_total_limit=3,
    report_to="swanlab",
    dataloader_num_workers=0,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    logging_steps=5,
    weight_decay=0.01,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)


def predict(sentence):
    model.eval()
    prompt = build_prompt(sentence)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    match = re.search(r'\d+', generated_text)
    if match:
        return int(match.group(0))
    return -1


def evaluate(df):
    preds, labels = [], []
    for sentence, label in zip(df['sentence'], df['label']):
        pred = predict(sentence)
        preds.append(pred)
        labels.append(label)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    return acc, f1


trainer.train()

dev_acc, dev_f1 = evaluate(df_dev)
print(f"Dev Acc: {dev_acc:.4f}, Dev F1: {dev_f1:.4f}")
swanlab.log({"dev_acc": dev_acc, "dev_f1": dev_f1})

test_acc, test_f1 = evaluate(df_test)
print(f"Test Acc: {test_acc:.4f}")
print(f"Test F1: {test_f1:.4f}")
swanlab.log({"test_acc": test_acc, "test_f1": test_f1})