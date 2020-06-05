import tensorflow as tf
import pickle
import numpy as np
from nltk.tokenize import word_tokenize
from model.auto_encoder import AutoEncoder
import re
# from data_utils import build_word_dict, build_word_dataset, batch_iter
BATCH_SIZE = 64


class QuestionEmbedding:
    def __init__(self, question, word_dict):
        self.question = question
        self.max_length = 50
        self.wordDict = word_dict

    def build_dataset(self):
        x = []
        for word in self.question.split():
            x.append(word)
        x = list(map(lambda w: self.wordDict.get(w, self.wordDict["<unk>"]), x))
        x = list(x[:self.max_length])
        x = x + (self.max_length - len(x)) * [self.wordDict["<pad>"]]
        return x

    def query(self):
        tf.reset_default_graph()
        model = AutoEncoder(self.wordDict, self.max_length)
        outputs = []

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            pre_trained_variables = [v for v in tf.global_variables()
                                     if (v.name.startswith("embedding") or v.name.startswith("birnn")) and "Adam" not in v.name]
            saver = tf.train.Saver(pre_trained_variables)
            ckpt = tf.train.get_checkpoint_state("checkpoint/")
            saver.restore(sess, ckpt.model_checkpoint_path)
            word2int = self.build_dataset()
            fake_batch = np.zeros((BATCH_SIZE, self.max_length))
            fake_batch[0] = word2int
            feed_dict = {model.x: fake_batch}
            encoder_outputs = sess.run([model.encoder_outputs], feed_dict=feed_dict)
            outputs.append(encoder_outputs[0])
        return outputs[0]
