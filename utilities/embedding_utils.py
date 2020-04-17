from utilities.utils import language_map
from gensim.models import Word2Vec


def get_word_vectors(language, dim=300):
    lang_full, lang_short = language_map(language)

    path = "trained_models/" + lang_full + "/"
    filename = lang_short + "_d" + str(dim) + ".model"

    model = Word2Vec.load(path + filename)

    return model.wv


def save_keyed_vectors(language, dim):
    lang_full, lang_short = language_map(language)

    path = "trained_models/" + lang_full + "/"
    filename = lang_short + "_d" + str(dim) + "_st.model"

    model = Word2Vec.load(path + filename)
    model.init_sims(replace=True)
    with open(path + lang_short + "_d" + str(dim) + "_st.bin", "wb") as f:
        model.wv.save_word2vec_format(f, binary=True)


save_keyed_vectors("en", 100)
