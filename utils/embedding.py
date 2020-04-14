from utils import language_map
import numpy as np
import pickle as p
import os


class Embedding:
    def __init__(self, lang, type="w2v", dim=100, max_vocab=50000):
        self.vector_dic = dict()
        self.filename = ''
        self.pickle_filename = ''

        lang_full, lang_short = language_map(lang)
        self.lang_full = lang_full
        self.lang_short = lang_short
        self.max_vocab = max_vocab
        self.dim = dim

        self.dir = "vector_models/" + lang_full

        if type == "w2v":
            self.get_w2v_embeddings()

        if type == "glove":
            self.get_glove_embeddings()

    def get_w2v_embeddings(self):
        self.pickle_filename = "w2v_embedding_d" + str(self.dim) + '.p'
        self.filename = self.lang_short + "wiki_20180420_" + str(self.dim) + "d.txt"
        if self.pickle_filename in os.listdir(self.dir):
            self.vector_dic = load_vector_dict(self.dir + "/" + self.pickle_filename)

        else:
            self.vector_dic = read_vector_file(self.filename, self.lang_full, self.max_vocab)
            self.save_pickle()

    def save_pickle(self):
        with open(self.dir + "/" + self.pickle_filename, "wb") as f:
            p.dump(self.vector_dic, f)

    def find_oov_word(self, oov_word):
        with open("vector_models/" + self.lang_full + "/" + self.filename) as f:
            _ = f.readline().split(' ')
            for line in f.readlines():
                line = line.rstrip()
                line = line.split(' ')
                if line[0] == oov_word:
                    vec = [float(x) for x in line[1:]]
                    vec = np.array(vec)
                    self.vector_dic[oov_word] = vec
                    self.save_pickle()
                    return vec

        return None

    def __getitem__(self, word):
        if word in self.vector_dic:
            return self.vector_dic[word]
        else:
            return self.find_oov_word(word)

    def get_glove_embeddings(self):
        self.pickle_filename = "glove_embedding_d" + str(self.dim) + '.p'

        self.filename = "glove.6B." + str(self.dim) + "d.txt"
        if self.pickle_filename in os.listdir(self.dir):
            self.vector_dic = load_vector_dict(self.dir + "/" + self.pickle_filename)

        else:
            self.vector_dic = read_vector_file(self.filename, self.lang_full, self.max_vocab)
            self.save_pickle()


def read_vector_file(filename, lang_full, max_vocab):
    vector_dic = {}

    with open("vector_models/" + lang_full + "/" + filename) as f:
        for i in range(max_vocab):
            line = f.readline().rstrip()
            line = line.split(' ')
            if len(line) < 10:
                continue

            word = line[0]
            vec = line[1:]

            if '\n' in line[-1]:
                line[-1] = line[-1][:-2]

            vec = [float(x) for x in vec]
            vector_dic[word] = np.array(vec)

        return vector_dic


def load_vector_dict(path):
    with open(path, "rb") as f:
        return p.load(f)
