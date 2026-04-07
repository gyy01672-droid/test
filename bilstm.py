import pandas as pd
from collections import Counter
from gensim.models import Word2Vec
for f in ['test.tsv']:
    tar = open('tar-' + f, 'w')
    tar.write('sentence\tlabel\n')
    for line in open(f).readlines():
        label = line[0]
        tar.write(line[2:].rstrip() + '\t' + label + '\n')

df_train = pd.read_csv('train.tsv', sep='\t')
df_dev = pd.read_csv('dev.tsv', sep='\t')
df_test = pd.read_csv('tar-test.tsv', sep='\t')

def tokenize(sentence):
    sentence = sentence.lower()
    sentence = sentence.split()
    return sentence

def vocab(sentence):
    counter = Counter()
    for sentences in sentence:
        tokens = tokenize(sentences)
        counter.update(tokens)
    vocab = {word: idx+1 for idx,(word,_) in enumerate(counter.items())}
    vocab["<pad>"]=0
    return vocab

train_vocab = vocab(df_train['sentence'])
dev_vocab = vocab(df_dev['sentence'])
test_vocab = vocab(df_test['sentence'])
