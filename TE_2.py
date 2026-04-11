import math
import re
from tabnanny import verbose

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
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
#整理数据

for f in ['test.tsv']:
    tar = open('tar-' + f, 'w')
    tar.write('sentence\tlabel\n')
    for line in open(f).readlines():
        label = line[0]
        tar.write(line[2:].rstrip() + '\t' + label + '\n')

df_train = pd.read_csv('train.tsv', sep='\t')
df_dev = pd.read_csv('dev.tsv', sep='\t')
df_test = pd.read_csv('tar-test.tsv', sep='\t')
#分词函数
def tokenize(sentence):
    sentence = str(sentence).lower().strip()
    return word_tokenize(sentence)
#创建词库函数
def vocab(sentence):
    counter = Counter()
    for sentences in sentence:
        tokens = tokenize(sentences)
        counter.update(tokens)
    vocab = {word: idx+1 for idx,(word,_) in enumerate(counter.items())}
    vocab["<pad>"]=0
    return vocab
#创建词库
train_vocab = vocab(df_train['sentence'])
dev_vocab = train_vocab
test_vocab = train_vocab
#字符编码
def encoder(sentence,vocab,max_len=100):
    tokens = tokenize(sentence)
    ids = [vocab.get(token,0) for token in tokens]
    if len(ids) < max_len:
        ids = ids + [0]*(max_len-len(ids))
    else:
        ids = ids[:max_len]
    return ids
train_ids = [encoder(s,train_vocab)for s in df_train['sentence']]
dev_ids = [encoder(s,train_vocab)for s in df_dev['sentence']]
test_ids = [encoder(s,test_vocab)for s in df_test['sentence']]

train_tokens = [tokenize(s)for s in df_train['sentence']]
dev_tokens = [tokenize(s)for s in df_dev['sentence']]
test_tokens = [tokenize(s)for s in df_test['sentence']]
#Word2Vec
all_tokens = train_tokens + dev_tokens + test_tokens
train_w2v_model = Word2Vec(sentences=all_tokens,vector_size=300,window=5,min_count=2,workers=4,epochs=20)
dev_w2v_model = train_w2v_model
test_w2v_model = train_w2v_model
#embedding
def embedding(vocab,w2v_model):
    vocab_size = len(vocab)
    embedding_dim = w2v_model.vector_size
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, idx in vocab.items():
        if word =="<pad>":
            continue
        if word in w2v_model.wv:
            embedding_matrix[idx] = w2v_model.wv[word]
        else:
            embedding_matrix[idx] = np.zeros(embedding_dim)
    return embedding_matrix
train_embedding = embedding(train_vocab,train_w2v_model)
dev_embedding = embedding(dev_vocab,dev_w2v_model)
test_embedding = embedding(test_vocab,test_w2v_model)
#定义位置信息
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100,dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
#统一定义超参数
hidden_dim = 512
dropout_rate = 0.1
batch_size = 32
learning_rate = 1e-5
epochs = 20
max_len = 100
mini_count = 2
nhead = 6
num_classes = 7
num_layers = 2
#transformer_encoder
class Transformer(nn.Module):
    def __init__(self,vocab_size,embedding_dim,num_classes,hidden_dim,num_layers,nhead,dropout_rate):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        self.embedding.weight.data.copy_(torch.tensor(train_embedding,dtype=torch.float))
        self.embedding.weight.requires_grad = True
        self.norm = nn.LayerNorm(embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim,max_len,dropout_rate)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim,nhead=nhead,dim_feedforward=hidden_dim*4,dropout=dropout_rate,batch_first=True,activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer,num_layers=num_layers)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(embedding_dim,num_classes)
    def forward(self,x):
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.norm(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class TextDataset(Dataset):
    def __init__(self,sentence,labels):
        self.sentence = torch.tensor(sentence,dtype=torch.long)
        self.labels = torch.tensor(labels,dtype=torch.long)
    def __len__(self):
        return len(self.sentence)
    def __getitem__(self, idx):
        return self.sentence[idx],self.labels[idx]
train_dataset = TextDataset(train_ids,df_train['label'].values)
dev_dataset = TextDataset(dev_ids,df_dev['label'].values)
test_dataset = TextDataset(test_ids,df_test['label'].values)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=32,shuffle=True)
dev_loader = torch.utils.data.DataLoader(dataset=dev_dataset,batch_size=32,shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=32,shuffle=False)
#swanlab初始化
run = swanlab.init(project="TE",experiment_name="TE",config={"hidden_dim":hidden_dim,"learning_rate":learning_rate,"epochs":epochs,"batch_size":batch_size,"max_len":max_len,"num_layers":num_layers,"nhead":nhead})
#定义训练函数
def train(model,train_loader,dev_loader,test_loader,epochs=epochs,lr=learning_rate,print_step=100,print_loss=True,device=torch.device('cpu')):
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=3,min_lr=1e-7)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x,y in train_loader:
            x,y = x.to(device),y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs,y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            optimizer.step()
            total_loss += loss.item()
        model.eval()
        all_preds,all_labels = [],[]
        with torch.no_grad():
            for x,y in dev_loader:
                x,y = x.to(device),y.to(device)
                outputs = model(x)
                preds = torch.argmax(outputs,dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        f1 = f1_score(all_labels,all_preds,average='macro')
        swanlab.log({"train_f1":f1})
        print(f"epoch{epoch+1}/{epochs} - Loss:{total_loss/len(train_loader):.4f} -Dev F1:{f1:.4f}")
        scheduler.step(total_loss / len(train_loader))
    model.eval()
    all_preds,all_labels = [],[]
    with torch.no_grad():
        for x,y in test_loader:
            x,y = x.to(device),y.to(device)
            outputs = model(x)
            preds = torch.argmax(outputs,dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    test_acc = accuracy_score(all_labels,all_preds)
    test_f1 = f1_score(all_labels,all_preds,average='macro')
    swanlab.log({"test_acc":test_acc,"test_f1":test_f1})
    print(f"\n测试集Accuracy:{test_acc:.4f} - F1:{test_f1:.4f}")

vocab_size = len(train_vocab)
embedding_dim = train_embedding.shape[1]
num_classes = 7
model = Transformer(vocab_size,embedding_dim,num_classes,hidden_dim,num_layers,nhead,dropout_rate)
device = torch.device('cpu')
train(model,train_loader,dev_loader,test_loader,device=device)