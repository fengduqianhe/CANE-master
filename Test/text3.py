import numpy as np
from tensorflow.contrib import learn
import random
import os

def load(text_path):
    text_file = open(text_path, 'rb').readlines()
    for a in range(0, len(text_file)):
        text_file[a] = str(text_file[a])
    return text_file


def load_text(text_file):
    vocab = learn.preprocessing.VocabularyProcessor(300)
    text = np.array(list(vocab.fit_transform(text_file)))
    print("text", text)
    num_vocab = len(vocab.vocabulary_)
    num_nodes = len(text)

    return text, num_vocab, num_nodes


if __name__ == "__main__":
    text_file = load("../datasets/cora/data.txt")
    text, num_vocab, num_nodes = load_text(text_file)
    print(text)
    print(text.shape)
    print(num_vocab)
    print(num_nodes)
