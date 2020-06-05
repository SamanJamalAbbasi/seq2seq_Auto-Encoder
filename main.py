import tensorflow as tf
from pre_train import train
from data_utils import build_word_dict, build_word_dataset
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from query import QuestionEmbedding

MAX_DOCUMENT_LEN = 50
file_dir = "data/wikismall-complex.txt"
data = open(file_dir, encoding='utf-8', errors='ignore').read().split('\n')

# Training all data and convert it to new Embedded representation :
print("\nBuilding dictionary..")
word_dict = build_word_dict()
print("Preprocessing dataset..")
train_x, train_y = build_word_dataset(word_dict, MAX_DOCUMENT_LEN)
embedded_data = train(train_x, train_y, word_dict)
# Test : Convert Question to embedded representation base on previous Trained model weights.
print(" Question: ")
tf.reset_default_graph()
query1 = "His Seven Stars Symphony features movements inspired by Douglas Fairbanks , Lilian Harvey ," \
         " Greta Garbo , Clara Bow , Marlene Dietrich , Emil Jannings and Charlie Chaplin in some of " \
         "their most famous film roles ."
with open("word_dict.pickle", "rb") as f:
    word_dict = pickle.load(f)

question_embedding = QuestionEmbedding(query1, word_dict)
embed_question = question_embedding.query()


def sum_embedded_words(encode_question):
    sent_embedded = []
    for word in encode_question:
        sent_embedded.append(sum(word).tolist())
    return sent_embedded


def similarity(x, q):
    scores = []
    for sentence in x:
        sentence = np.reshape(sentence, [1, 512])
        scores.append(cosine_similarity(sentence, q))
    return scores


question = sum_embedded_words(embed_question[:1])
question = np.reshape(question, [1, 512])
score = similarity(embedded_data, question)
max_similarity = score.index(max(score))
most_similar_answer = data[max_similarity]
print(max(score))
print(max_similarity)
print(most_similar_answer)
