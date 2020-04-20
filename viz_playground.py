from utilities.utils import visualize_embeddings, language_map
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.keyedvectors import FastTextKeyedVectors
from gensim.models import FastText
import os


def visualize_language(model_name):
    path = "data/vector_models/" + model_name

    if "ft" in model_name:
        wv = FastTextKeyedVectors.load(path)
    else:
        wv = KeyedVectors.load_word2vec_format(path, binary=True)

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

    if "ft" in model_name:
        words.extend(["skjafk", "fjhfkja", "fhsdhlf"])

    visualize_embeddings(wv, words)
    print(len(wv.vocab))
    dir(wv)


dirs = os.listdir("data/vector_models")
for i, model_name in enumerate(dirs):
    print(i, model_name)

ind = int(input("Model:"))
model_name = dirs[ind]
print(model_name)

visualize_language(model_name)