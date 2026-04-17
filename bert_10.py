import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import math
import re
from tabnanny import verbose
from transformers import BertTokenizer,BertForSequenceClassification
from torch.optim import AdamW
import numpy as np
import pandas as pd
import swanlab
import torch
from collections import Counter
from gensim.models import Word2Vec
from torch import nn, unique
from torch.utils.data import Dataset
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
#整理数据

def readdata(filename):
    df=pd.read_csv(filename,sep=None,engine='python')
    df.columns = [c.strip() for c in df.columns]
    df = df[['Text','Label']]
    df['Text'] = df['Text'].astype(str).str.lower().str.strip()
    df['Label'] = df['Label'].astype(str).str.strip()
    return df
df_train = readdata('train.csv')
df_test = readdata('test.csv')
unique_labels = sorted(df_train['Label'].unique())
label_map = {label: idx for idx, label in enumerate(unique_labels)}
df_train['label'] = df_train['Label'].map(label_map).astype(int)
df_test['label'] = df_test['Label'].map(label_map).fillna(0).astype(int)
#划分验证集
try:
    df_train,df_dev = train_test_split(df_train,test_size=0.2,random_state=42,stratify=df_train['label'])
except ValueError as e:
    df_train,df_dev = train_test_split(df_train,test_size=0.2,random_state=42,stratify=None)
#统一定义超参数
batch_size = 32
learning_rate = 2e-5
epochs = 20
max_len = 100
num_classes = 7
#tokenizer and encoder
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def bert_encoder(sentences,max_len=100):
    return tokenizer(sentences.tolist(),padding=True,truncation=True,max_length=max_len,return_tensors='pt')
train_encodings = bert_encoder(df_train['Text'])
dev_encodings = bert_encoder(df_dev['Text'])
test_encodings = bert_encoder(df_test['Text'])

class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels,dtype=torch.long)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key:val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

train_dataset = TextDataset(train_encodings,df_train['label'].values)
dev_dataset = TextDataset(dev_encodings,df_dev['label'].values)
test_dataset = TextDataset(test_encodings,df_test['label'].values)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=32,shuffle=True)
dev_loader = torch.utils.data.DataLoader(dataset=dev_dataset,batch_size=32,shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=32,shuffle=False)
#swanlab初始化
run = swanlab.init(project="bert_10",experiment_name="bert_10")
#定义训练
model = BertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=num_classes)
for name, param in model.named_parameters():
    if "encoder.layer.10" in name or "encoder.layer.11" in name or "encoder.layer.9" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False
device = torch.device("cuda")
model.to(device)
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids,attention_mask=attention_mask,labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dev_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids,attention_mask=attention_mask,labels=labels)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"epoch{epoch+1}|Loss {total_loss/len(train_loader):.4f}|Dev F1 = {f1:.4f}")
    swanlab.log({"dev_f1":f1})
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids,attention_mask=attention_mask,labels=labels)
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
test_acc = accuracy_score(all_labels, all_preds)
test_f1 = f1_score(all_labels, all_preds, average='macro')
print(f"test accuracy: {test_acc}")
print(f"test f1: {test_f1:.4f}")
swanlab.log({"test_acc":test_acc,"test_f1":test_f1})