import argparse

from models import W2V, Generator, preproc_predict, preproc, create_dicts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--model', type=str, help='Path to load the model')
    parser.add_argument('--word2vec', type=str, help='Path to load word2vec')
    parser.add_argument('--prefix', default=None, type=str, help='Input to model')
    parser.add_argument('--length', default=1, type=int,  help='Textgen length')
    arguments = parser.parse_args()
    text = None
    if arguments.prefix:
        text=preproc_predict(arguments.prefix)

    w2v = W2V.load_model(arguments.word2vec)
    gen = Generator.load_model(arguments.model)
    print(gen.predict(text, arguments.length))
