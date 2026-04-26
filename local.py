import os
import torch
import pandas as pd
from transformers import LlamaTokenizer, LlamaForCausalLM
from sklearn.metrics import accuracy_score, f1_score
import swanlab

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"


def readdata(filename):
    df = pd.read_csv(filename, sep='\t', engine='python')
    df.columns = [c.strip() for c in df.columns]
    df = df[['Text', 'Label']]
    df.columns = ['sentence', 'Label']
    df['sentence'] = df['sentence'].astype(str).str.replace('á', ' ').str.strip()
    df['Label'] = df['Label'].astype(str).str.strip()
    return df


df_train = readdata('train.csv')
df_test = readdata('test.csv')

unique_labels = sorted(df_train['Label'].unique())
label_map = {label: idx for idx, label in enumerate(unique_labels)}

tokenizer = LlamaTokenizer.from_pretrained(
    '/root/autodl-fs/llama-3.1-8b-instruct',
    use_fast=False,
    local_files_only=True
)
tokenizer.pad_token = tokenizer.eos_token

model = LlamaForCausalLM.from_pretrained(
    '/root/autodl-fs/llama-3.1-8b-instruct',
    local_files_only=True,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map='auto'
)
model.eval()


def build_prompt(sentence):
    labels_str = ", ".join(unique_labels)
    prompt = f"""### Instruction:
Classify the emotion of the sentence into one of these categories: {labels_str}. Respond with only the label name.

### Input:
{sentence}

### Response:
"""
    return prompt


def predict(sentence):
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

    for label in unique_labels:
        if label.lower() in generated_text.lower():
            return label
    return unique_labels[0]


def evaluate(df, max_samples=None):
    preds, labels = [], []
    test_df = df if max_samples is None else df.head(max_samples)

    for i, row in test_df.iterrows():
        pred = predict(row['sentence'])
        preds.append(pred)
        labels.append(row['Label'])

        if (i + 1) % 10 == 0:
            print(f"已处理 {i + 1}/{len(test_df)} 条")

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')

    return acc, f1


run = swanlab.init(project="model", experiment_name="zero_shot")

print("=" * 50)
print("零样本学习 (Zero-Shot)")
print("=" * 50)

acc, f1 = evaluate(df_test, max_samples=50)

print(f"\n结果: Acc = {acc:.4f}, F1 = {f1:.4f}")

swanlab.log({"test_acc": acc, "test_f1": f1})
