from utilities.utils import language_map
from gensim.models import Word2Vec, FastText
import os


def get_word_vectors(language, dim=300):
    lang_full, lang_short = language_map(language)

    path = "trained_models/" + lang_full + "/"
    filename = lang_short + "_d" + str(dim) + ".model"

    model = Word2Vec.load(path + filename)

    return model.wv


def save_keyed_vectors(model_path, model_name):
    save_path = "data/vector_models/"
    filename = model_name + ".model"

    if "ft" in model_name:
        print("Binning fasttext model!")
        model = FastText.load(model_path + filename)
    else:
        model = Word2Vec.load(model_path + filename)

    model.init_sims(replace=True)
    with open(save_path + model_name + ".bin", "wb") as f:
        model.wv.save_word2vec_format(f, binary=True)


def bin_all():
    DIR = "trained_models/"
    for lang in os.listdir(DIR):
        for file in os.listdir(DIR + lang):
            ext = os.path.splitext(file)
            if ext[-1] == ".model":
                model_path = DIR + lang + "/"
                model_name = str(ext[0])

                print("Binning", model_name)
                try:
                    save_keyed_vectors(model_path, model_name)
                except:
                    print("Failed to bin", model_name)


bin_all()
