from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from utilities.utils import language_map
from gensim.models import KeyedVectors
from gensim.models.keyedvectors import FastTextKeyedVectors
import os
import numpy as np

en_words = ["monkey", "dog", "cat", "cow", "car", "bike", "taxi", "cab", "airplane", "plane", "train",
            "one", "two", "three", "four", "five", "six",
            "amsterdam", "london", "berlin", "rotterdam", "amsterdam",
            "netherlands", "germany", "england"]

nl_words = ["aap", "hond", "kat", "koe", "auto", "fiets", "taxi", "vliegtuig", "trein",
            "een", "twee", "drie", "vier", "vijf", "zes",
            "amsterdam", "london", "berlin", "rotterdam", "manchester",
            "nederland", "engeland", "duitsland"]

ru_words = []

all_words = {"nl": nl_words, "en": en_words, "ru": ru_words}


def get_model_name():
    dirs = os.listdir("data/vector_models")
    dirs.sort()
    for i, model_name in enumerate(dirs):
        print(i, "\t", model_name)

    ind = int(input("Model:"))
    if ind < len(dirs):
        model_name = dirs[ind]
        return model_name
    else:
        print("Index not valid!")
        return get_model_name()


def visualize_words(wv, words):
    pca = PCA(n_components=2)

    X = wv[words]
    result = pca.fit_transform(X)

    plt.scatter(result[:, 0], result[:, 1])
    for i, word in enumerate(words):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))

    plt.show()


def visualize_language():
    model_name = get_model_name()
    path = "data/vector_models/" + model_name

    if "ft" in model_name:
        wv = FastTextKeyedVectors.load(path)
    else:
        wv = KeyedVectors.load_word2vec_format(path, binary=True)

    print("Vocab size:", len(wv.vocab))

    words = [""]
    if "en" in model_name:
        words = en_words

    if "nl" in model_name:
        words = nl_words

    visualize_words(wv, words)


def get_subplot_for_data(lang="en"):
    lang_full, lang_short = language_map(lang)
    fig = plt.figure()

    plot_labels = {"w2v": "Word2Vec", "ft": "FastText",
                   "cbow": "CBOW", "sg": "Skip-Gram"}

    for i, type in enumerate(["w2v", "ft"]):
        for j, hp in enumerate(["cbow", "sg"]):
            print(type, hp)

            # First word2vec
            model_name = type + "_" + lang + "_d100_" + hp + "_st.bin"
            path = "data/vector_models/" + model_name

            if type == "ft":
                wv = FastTextKeyedVectors.load(path)
            else:
                wv = KeyedVectors.load_word2vec_format(path, binary=True)

            words = all_words[lang]
            pca = PCA(n_components=2)

            # pca.fit(wv[wv.vocab])

            X = wv[words]
            X -= np.mean(X, axis=0)
            X /= np.var(X, axis=0)
            result = pca.fit_transform(X)

            # Start subplot
            subplot_num = i * 2 + (j + 1)
            axis = fig.add_subplot(2, 2, subplot_num)

            axis.scatter(result[:, 0], result[:, 1])
            for k, word in enumerate(words):
                axis.annotate(word, xy=(result[k, 0], result[k, 1]), size=7)

            axis.title.set_text(lang_full.capitalize() + " - " + plot_labels[type] + " using " + plot_labels[hp])

    plt.show()


print("Starting subplots")
get_subplot_for_data("nl")


