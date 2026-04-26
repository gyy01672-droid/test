import os
import time
import pandas as pd
import requests
from sklearn.metrics import accuracy_score, f1_score

API_KEY = ""
BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "google/gemini-3-flash-preview"


def readdata(filename):
    df = pd.read_csv(filename, sep='\t', engine='python')
    df.columns = [c.strip() for c in df.columns]
    df = df[['Text', 'Label']]
    df.columns = ['sentence', 'Label']
    df['Label'] = df['Label'].astype(str).str.strip()
    return df


df_train = readdata('train.csv')
df_test = readdata('test.csv')

unique_labels = sorted(df_train['Label'].unique())
label_map = {label: idx for idx, label in enumerate(unique_labels)}


def call_llm(prompt):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 10
    }
    try:
        response = requests.post(
            f"{BASE_URL}/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        result = response.json()
        return result['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"API错误: {e}")
        return ""


def build_prompt(sentence):
    labels_str = ", ".join(unique_labels)
    prompt = f"""请将以下句子分类为以下情感类别之一：{labels_str}

句子：{sentence}

情感类别："""
    return prompt


def extract_label(text):
    text = text.lower().strip()
    for label in unique_labels:
        if label.lower() in text:
            return label
    return unique_labels[0]


def evaluate(df, max_samples=None):
    preds, labels = [], []
    test_df = df if max_samples is None else df.head(max_samples)

    for i, row in test_df.iterrows():
        sentence = row['sentence']
        true_label = row['Label']

        prompt = build_prompt(sentence)
        raw_output = call_llm(prompt)
        pred_label = extract_label(raw_output)

        preds.append(pred_label)
        labels.append(true_label)

        if (i + 1) % 10 == 0:
            print(f"已处理 {i + 1}/{len(test_df)} 条")
        time.sleep(0.5)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')

    return {
        'accuracy': acc,
        'f1_macro': f1,
        'predictions': preds,
        'labels': labels
    }


result = evaluate(df_test, max_samples=50)

print(f"\n结果: Acc = {result['accuracy']:.4f}, F1 = {result['f1_macro']:.4f}")