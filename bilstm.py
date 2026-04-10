import numpy as np
import pandas as pd
import swanlab
from collections import Counter
from gensim.models import Word2Vec
from sympy.printing.pytorch import torch
from torch import nn
from torch.utils.data import Dataset
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

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
    sentence = sentence.lower()
    sentence = sentence.split()
    return sentence
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
def encoder(sentence,vocab,max_len=20):
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
train_w2v_model = Word2Vec(sentences=train_tokens,vector_size=100,window=5,min_count=1,workers=4)
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
#统一定义超参数
hidden_dim = 128
dropout_rate = 0.7
batch_size = 32
learning_rate = 0.0005
epochs = 5
max_len = 20
#lstm
class LSTM(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hideen_dim,num_classes,embedding_matrix,dropout_rate):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        self.embedding.weight.data.copy_(torch.tensor(embedding_matrix,dtype=torch.float))
        self.lstm = nn.LSTM(embedding_dim,hideen_dim,batch_first=True,bidirectional=True)
        self. dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hideen_dim*2,num_classes)
    def forward(self,x):
        x = self.embedding(x)
        out,(h_n,c_n) = self.lstm(x)
        forward_last = h_n[-2]
        backward_last = h_n[-1]
        out = torch.cat((forward_last,backward_last),1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
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
run = swanlab.init(project="bilstm",experiment_name="bilstm",config={"hidden_dim":hidden_dim,"learning_rate":learning_rate,"epochs":epochs,"batch_size":batch_size,"max_len":20})
#定义训练函数
def train(model,train_loader,dev_loader,test_loader,epochs=epochs,lr=learning_rate,print_step=100,print_loss=True,device=torch.device('cpu')):
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
num_classes = 2
model = LSTM(vocab_size,embedding_dim,hidden_dim,num_classes,train_embedding,dropout_rate)
device = torch.device('cpu')
train(model,train_loader,dev_loader,test_loader,device=device)