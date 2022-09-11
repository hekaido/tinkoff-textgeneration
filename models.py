import pickle
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from gensim.corpora import Dictionary
from gensim.models import Word2Vec
from gensim.utils import tokenize
from tqdm import tqdm


def preproc_predict(line):
    clean = []
    clean.append(list(tokenize(line, lowercase=True, deacc=True)))
    if len(clean[0]) < 8:
        seq = clean[0]
        while len(seq) < 8:
            seq.insert(0,'pad')
        clean[0]=seq
    return clean


def preproc(lines, sentences):
    clean = []
    for line in lines:
        clean.append(list(tokenize(line, lowercase=True, deacc=True)))
    clean = [line for line in clean if line and len(line) >= 15]
    clean = clean[:sentences]
    clean[0].insert(0,'pad')
    return clean


def create_dicts(sentences):
    dic = Dictionary()
    for line in sentences:
        dic.add_documents([line])
    py_dict = dic.token2id
    rev_py_dict = {v: k for k, v in py_dict.items()}
    return py_dict, rev_py_dict


class W2V:
    def __init__(
            self, corpus: Union[str, list], vector_size, window, min_count, workers
    ):
        self.model = Word2Vec(
            corpus,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
        )
        self.vector_size = vector_size
        self.corpus = corpus

    def __getitem__(self, item):
        return self.model.wv[item]

    def __len__(self):
        return self.vector_size

    def train(self, epochs):
        self.model.train(self.corpus, total_examples=len(self.corpus), epochs=epochs)

    def save_model(self, path):
        if not path.endswith("pkl"):
            raise ValueError("Extension must be pkl")
        with open(f"{path}", "wb") as f:
            pickle.dump(self, file=f)

    @staticmethod
    def load_model(path):
        if not path.endswith("pkl"):
            raise ValueError("Extension must be pkl")
        with open(f"{path}", "rb") as f:
            model = pickle.load(file=f)
        return model


class Generator:
    def __init__(self, w2v, edge, py_dict, rev_py_dict):
        self.hidden = nn.Sequential(
            nn.Linear(
                in_features=edge * len(w2v), out_features=len(py_dict), bias=True
            )
        )
        self.optimizer = torch.optim.Adam(self.hidden.parameters())
        self.loss_f = nn.CrossEntropyLoss()
        self.w2v = w2v
        self.edge = edge
        self.py_dict = py_dict
        self.rev_py_dict = rev_py_dict

    def fit(self, text, epochs):
        self.hidden.train()
        for epoch in range(epochs):
            for seq in tqdm(text):

                cur_seq = self.w2v[seq[: self.edge]]
                mean = cur_seq.mean()
                cur_seq /= mean
                cur_seq = torch.tensor(cur_seq)
                self.optimizer.zero_grad()

                pred = self.hidden(cur_seq.flatten())
                target = torch.tensor(
                    np.zeros(len(self.py_dict), dtype=np.float64), dtype=torch.float64
                )
                target[self.py_dict[seq[self.edge]]] = 1.0
                loss = self.loss_f(pred, target)
                loss.backward()

                self.optimizer.step()

                for i in range(self.edge + 1, len(seq) - 1):
                    cur_seq *= mean
                    next_word = torch.tensor([self.w2v[seq[i]]], dtype=torch.float64)
                    cur_seq = cur_seq[1:]
                    cur_seq = torch.cat([cur_seq, next_word], axis=0)
                    mean = torch.mean(cur_seq)
                    cur_seq /= mean.item()

                    self.optimizer.zero_grad()
                    pred = self.hidden(cur_seq.flatten().float())
                    target = torch.tensor(
                        np.zeros(len(self.py_dict), dtype=np.float64),
                        dtype=torch.float64,
                    )
                    target[self.py_dict[seq[i]]] = 1.0
                    loss = self.loss_f(pred, target)
                    self.optimizer.step()

    def predict(self, text, length):
        self.hidden.eval()
        generated_text = []
        for i in range(length):
            seq = self.w2v[text[0][i: self.edge + i]]
            seq = torch.tensor(seq)
            seq /= torch.mean(seq).item()
            with torch.no_grad():
                preds = self.hidden(seq.flatten())
                preds, index = torch.sort(preds)[-10:]
                index=index.numpy()
                index=np.random.choice(index,1)[0]
                text[0].append(self.rev_py_dict[index])
                generated_text.append(self.rev_py_dict[index])
        return generated_text

    def save_model(self, path):
        if not path.endswith("pkl"):
            raise ValueError("Extension must be pkl")
        with open(f"{path}", "wb") as f:
            pickle.dump(self, file=f)

    @staticmethod
    def load_model(path):
        if not path.endswith("pkl"):
            raise ValueError("Extension must be pkl")
        with open(f"{path}", "rb") as f:
            model = pickle.load(file=f)
        return model
