from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from utilities.utils import language_map
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.keyedvectors import FastTextKeyedVectors
from gensim.models import FastText
import os


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
    X = wv[wv.vocab]

    pca = PCA(n_components=2)
    pca.fit(X)

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
        words = ["monkey", "dog", "cat", "cow", "car", "bike", "taxi", "cab", "airplane", "plane", "train",
                 "one", "two", "three", "four", "five", "six",
                 "amsterdam", "london", "berlin", "rotterdam", "amsterdam",
                 "netherlands", "germany", "england"]

    if "nl" in model_name:
        words = ["aap", "hond", "kat", "koe", "auto", "fiets", "taxi", "vliegtuig", "trein",
                 "een", "twee", "drie", "vier", "vijf", "zes", ".", "<SOS>", "<EOS>",
                 "amsterdam", "london", "berlin", "rotterdam", "manchester"]

    visualize_words(wv, words)


visualize_language()


