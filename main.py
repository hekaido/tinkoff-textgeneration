from gensim.models import Word2Vec
from gensim.utils import tokenize
from gensim.corpora import Dictionary
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import  Union

def preproc(lines, sentences):
    clean = []
    for line in lines:
        clean.append(list(tokenize(line, lowercase=True, deacc=True)))
    clean = [line for line in clean if line and len(line) >= 15]
    clean = clean[:sentences]
    return clean


def create_dicts(sentences):
    dic = Dictionary()
    for line in sentences:
        dic.add_documents([line])
    py_dict = dic.token2id
    rev_py_dict = {v: k for k, v in py_dict.items()}
    return (py_dict, rev_py_dict)


class word2vec():
    def __init__(self, corpus: Union[str, list], vector_size, window, min_count, workers):
        self.model = Word2Vec(
            sentences=corpus,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers
        )
        self.vector_size = vector_size
        self.corpus=corpus

    def __getitem__(self, item):
        return self.model.wv[item]

    def __len__(self):
        return self.vector_size

    def train(self, epochs):
        self.model.train(sentences=self.corpus, total_examples=len(self.corpus), epochs=epochs)


class generator(nn.Module):
    def __init__(self, w2v):
        self.hidden = nn.Sequential(
            nn.Linear(in_features=edge * 128, out_features=len(py_dict), bias=True),
            nn.Softmax(dim=0)
        )
        self.optimizer = torch.optim.Adam(self.hidden.parameters())
        self.loss_f = nn.CrossEntropyLoss()
        self.w2v = w2v

    def fit(self, text, epochs, edge, pydict):
        self.hidden.train()
        for epoch in range(epochs):
            for seq in tqdm(text):

                cur_seq = self.w2v[seq[:edge]]
                mean = cur_seq.mean()
                cur_seq /= mean
                cur_seq = torch.tensor(cur_seq)
                self.optimizer.zero_grad()

                pred = self.hidden(cur_seq.flatten())
                target = torch.tensor(np.zeros(len(py_dict), dtype=np.float64), dtype=torch.float64)
                target[py_dict[seq[edge]]] = 1.0
                loss = self.loss_f(pred, target)
                loss.backward()

                self.optimizer.step()

                for i in range(edge + 1, len(seq) - 1):
                    cur_seq *= mean
                    next_word = torch.tensor([self.w2v[seq[i]]], dtype=torch.float64)
                    cur_seq = cur_seq[1:]
                    cur_seq = torch.cat([cur_seq, next_word], axis=0)
                    mean = torch.mean(cur_seq)
                    cur_seq /= mean[0]

                    self.optimizer.zero_grad()
                    pred = self.hidden(cur_seq.flatten().float())
                    target = torch.tensor(np.zeros(len(py_dict), dtype=np.float64), dtype=torch.float64)
                    target[py_dict[seq[i]]] = 1.0
                    loss = self.loss_f(pred, target)

                    self.optimizer.step()
        self.save_model()

    def predict(self, text, length):
        self.hidden.eval()
        for i in length:
            seq = self.w2v[text[i:edge + i]]
            seq /= seq.mean
            seq = torch.tensor(seq)
            with torch.no_grad():
                preds = self.hidden(seq)
                index = torch.argmax(preds)
                print(rev_py_dict[index])
                text.append(rev_py_dict[index])

    def safe_model(self):
        torch.save(self.hidden.state_dict(), 'saves.pt')

    def load_model(self):
        self.hidden = self.hidden.load_state_dict(torch.load('saves.pt'))


# preproc
file = open('oblomov.txt', 'r', encoding='utf8')
lines = file.readlines()

clean = []

clean = preproc(lines, 1000)

py_dict, rev_py_dict = create_dicts(clean)

w2v = word2vec(sentences=clean, vector_size=64, window=5, min_count=1, workers=4)

w2v.train(sentences=clean, epochs=5)

clean = np.array([np.array(line, dtype=object) for line in clean], dtype=object)

edge = 8
epochs = 1

gen = generator(w2v)

gen.fit(text=clean, epochs=epochs, edge=edge, py_dict=py_dict)
length = 10
