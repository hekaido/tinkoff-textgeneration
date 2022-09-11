import numpy as np
import argparse
import os

from models import W2V, Generator, create_dicts, preproc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--input-dir', type=str, help='Dir data path')
    parser.add_argument('--model', type=str, help='Model save path')
    parser.add_argument('--word2vec', type=str, help='Word2vec save path')
    arguments=parser.parse_args()
    lines=[]
    if arguments.input_dir:
        for file in os.listdir(arguments.input_dir):
            file = open(f"{arguments.input_dir}/{file}", "r", encoding="utf8")
            line=file.readlines()
            for lin in line:
                lines.append(lin)
    else:
        lines=str(input())
    clean = []

    clean = preproc(lines, 1000)

    py_dict, rev_py_dict = create_dicts(clean)

    w2v = W2V(corpus=clean, vector_size=64, window=5, min_count=1, workers=4)
    w2v.train(epochs=5)
    w2v.save_model(arguments.word2vec)
    clean = np.array([np.array(line, dtype=object) for line in clean], dtype=object)
    edge = 8
    epochs = 1
    gen = Generator(
        w2v=w2v,
        edge=edge,
        py_dict=py_dict,
        rev_py_dict=rev_py_dict
    )
    gen.fit(text=clean, epochs=epochs)
    gen.save_model(arguments.model)
