from utilities.utils import visualize_embeddings, language_map
from gensim.models import Word2Vec


def visualize_language(lang):
    lang_full, lang_short = language_map(lang)

    model = Word2Vec.load("trained_models/" + lang_full + "/" + lang_short + "_d100_st.model")
    model.init_sims(replace=True)

    words = [""]
    if lang_short == "en":
        words = ["monkey", "dog", "cat", "cow", "car", "bike", "taxi", "cab", "airplane", "plane", "train",
                 "one", "two", "three", "four", "five", "six",
                 "amsterdam", "london", "berlin", "rotterdam", "amsterdam",
                 "netherlands", "germany", "england"]
    if lang_short == "nl":
        words = ["aap", "hond", "kat", "koe", "auto", "fiets", "taxi", "vliegtuig", "trein",
                 "een", "twee", "drie", "vier", "vijf", "zes", ".", "<SOS>", "<EOS>",
                 "amsterdam", "london", "berlin", "rotterdam", "manchester"]

    visualize_embeddings(model, words)


visualize_language("dutch")