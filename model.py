import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer,AutoModelForCausalLM,TrainingArguments,Trainer
from peft import LoraConfig,get_peft_model
from transformers import LlamaTokenizer,LlamaForCausalLM
import swanlab
import re
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
#读取数据
for f in ['test.tsv']:
    tar = open('tar-' + f, 'w')
    tar.write('sentence\tlabel\n')
    for line in open(f).readlines():
        label = line[0]
        tar.write(line[2:].rstrip() + '\t' + label + '\n')

df_train = pd.read_csv('train.tsv', sep='\t')
df_dev = pd.read_csv('dev.tsv', sep='\t')
df_test = pd.read_csv('tar-test.tsv', sep='\t')
#prompt工程
def build_prompt(sentence,label=None):
    prompt = f"""###Instruction:
    Classify the emotion of the sentence.
    ###Input:
    {sentence}
    ###Response:
    The label is {label}
"""
    if label is not None:
        prompt += str(label)
    return prompt
train_texts = [build_prompt(s,l)for s,l in zip(df_train['sentence'],df_train['label'])]
train_dataset = Dataset.from_dict({'text': train_texts})
dev_texts = [build_prompt(s,l)for s,l in zip(df_dev['sentence'],df_dev['label'])]
model_name="Qwen/Qwen2-7B-Instruct"
#tokenizer
tokenizer = LlamaTokenizer.from_pretrained('/root/autodl-fs/llama-3.1-8b-instruct', use_fast=False, local_files_only=True)
#padding
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

        # 用 tokenizer 找 Response token位置
        response_tokens = tokenizer("###Response:", add_special_tokens=False)["input_ids"]

        # 在 input_ids 中找子序列
        def find_sublist(lst, sub):
            for j in range(len(lst) - len(sub)):
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
model = LlamaForCausalLM.from_pretrained('/root/autodl-fs/llama-3.1-8b-instruct', local_files_only=True,dtype=torch.float16,low_cpu_mem_usage=True, device_map='auto')
#配置lora
lora_config = LoraConfig(r=16,lora_alpha=32,target_modules=["q_proj","k_proj","v_proj","o_proj"],lora_dropout = 0.1,bias="none",task_type="CAUSAL_LM")
model = get_peft_model(model,lora_config)
model.print_trainable_parameters()
#swanlab
run = swanlab.init(project="model",experiment_name="model")
#训练参数
training_args = TrainingArguments(
    output_dir="./llama-lara",
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
)
#训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)
def predict(sentence):
    prompt = build_prompt(sentence)
    inputs = tokenizer(prompt,return_tensors="pt").to('cuda')
    outputs = model.generate(**inputs,max_new_tokens=20)
    text = tokenizer.decode(outputs[0],skip_special_tokens=True)
    return text
#提取标签
def extract_label(text):
    match = re.search(r"label is (\d+)",text)
    if match:
        return int(match.group(1))
    else:
        return -1
def evaluate(df):
    model.eval()
    preds,labels = [],[]
    for sentence,label in zip(df['sentence'],df['label']):
        output = predict(sentence)
        pred = extract_label(output)
        preds.append(pred)
        labels.append(label)
    acc = accuracy_score(labels,preds)
    f1 = f1_score(labels,preds,average='macro')
    return acc,f1

trainer.train()
dev_acc,dev_f1 = evaluate(df_dev)
swanlab.log({"dev_f1": dev_f1})
#测试
model.eval()
all_preds = []
all_labels = []
for sentence, label in zip(df_test["sentence"], df_test["label"]):
    output = predict(sentence)
    pred = extract_label(output)
    all_preds.append(pred)
    all_labels.append(label)
test_acc = accuracy_score(all_labels, all_preds)
test_f1 = f1_score(all_labels, all_preds, average="macro")
print(f"Test Acc: {test_acc:.4f}")
print(f"Test F1: {test_f1:.4f}")
swanlab.log({"test_acc": test_acc,"test_f1": test_f1})