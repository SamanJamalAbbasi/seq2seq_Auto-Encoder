import os
# import wget
# import tarfile
import re
from nltk.tokenize import word_tokenize
import collections
import pickle
import numpy as np


file_dir = "data/wikismall-complex.txt"
input_text = open(file_dir, encoding='utf-8', errors='ignore').read().split('\n')
TRAIN_PATH = file_dir
TEST_PATH = file_dir


def clean_str(text):
    text = re.sub(r"[^A-Za-z0-9(),!?\'\`\"]", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text


def build_word_dict():
    if not os.path.exists("word_dict.pickle"):
        contents = input_text
        words = list()
        for content in contents:
            for word in word_tokenize(clean_str(content)):
                words.append(word)

        word_counter = collections.Counter(words).most_common()
        word_dict = dict()
        word_dict["<pad>"] = 0
        word_dict["<unk>"] = 1
        word_dict["<s>"] = 2
        word_dict["</s>"] = 3
        for word, count in word_counter:
            if count > 1:
                word_dict[word] = len(word_dict)

        with open("word_dict.pickle", "wb") as f:
            pickle.dump(word_dict, f)

    else:
        with open("word_dict.pickle", "rb") as f:
            word_dict = pickle.load(f)

    return word_dict


def build_word_dataset(word_dict, document_max_len):
    df = input_text
    x = list(map(lambda d: word_tokenize(clean_str(d)), df))
    x = list(map(lambda d: list(map(lambda w: word_dict.get(w, word_dict["<unk>"]), d)), x))
    x = list(map(lambda d: d[:document_max_len], x))
    x = list(map(lambda d: d + (document_max_len - len(d)) * [word_dict["<pad>"]], x))
    y = list(map(lambda d: d, list(df)))

    return x, y


def batch_iter(inputs, outputs, batch_size, num_epochs):
    inputs = np.array(inputs)
    outputs = np.array(outputs)

    num_batches_per_epoch = (len(inputs) - 1) // batch_size + 1
    for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(inputs))
            yield inputs[start_index:end_index], outputs[start_index:end_index]


# if __name__ == "__main__":
#     MAX_DOCUMENT_LEN = 100
#     word_dict = build_word_dict()
#     train_x, train_y = build_word_dataset("train", word_dict, MAX_DOCUMENT_LEN)
#     print("END")
