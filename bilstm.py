import numpy as np
import pandas as pd
from collections import Counter
from gensim.models import Word2Vec
from sympy.printing.pytorch import torch
from torch import nn

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
dev_vocab = vocab(df_dev['sentence'])
test_vocab = vocab(df_test['sentence'])
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
dev_ids = [encoder(s,dev_vocab)for s in df_dev['sentence']]
test_ids = [encoder(s,test_vocab)for s in df_test['sentence']]

train_tokens = [tokenize(s)for s in df_train['sentence']]
dev_tokens = [tokenize(s)for s in df_dev['sentence']]
test_tokens = [tokenize(s)for s in df_test['sentence']]
#Word2Vec
train_w2v_model = Word2Vec(sentences=train_tokens,vector_size=100,window=5,min_count=1,workers=4)
dev_w2v_model = Word2Vec(sentences=dev_tokens,vector_size=100,window=5,min_count=1,workers=4)
test_w2v_model = Word2Vec(sentences=test_tokens,vector_size=100,window=5,min_count=1,workers=4)
#embedding
def embedding(vocab,w2v_model):
    vocab_size = len(vocab)
    embedding_dim = w2v_model.vector_size
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, idx in vocab.items():
        embedding_matrix[idx] = w2v_model[word]
    return embedding_matrix
train_embedding = embedding(train_vocab,train_w2v_model)
dev_embedding = embedding(dev_vocab,dev_w2v_model)
test_embedding = embedding(test_vocab,test_w2v_model)
#lstm
class LSTM(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hideen_dim,num_classes,embedding_matrix):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        self.embedding.weight.data.copy_(torch.tenser(embedding_matrix,dtype=torch.float))
        self.lstm = nn.LSTM(embedding_dim,hideen_dim,batch_first=True,bidirectional=True)
        self.fc = nn.Linear(hideen_dim*2,num_classes)
        def forward(self,x):
            x = self.embedding(x)
            out,(h_n,c_n) = self.lstm(x)
            forward_last = h_n[-2]
            backward_last = h_n[-1]
            out = torch.cat((forward_last,backward_last),1)
            out = self.fc(out)
            return out
