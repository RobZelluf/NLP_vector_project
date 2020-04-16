from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_multiple_whitespaces, strip_punctuation
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def visualize_embeddings(model, words):
    X = model[model.wv.vocab]

    pca = PCA(n_components=2)
    pca.fit(X)

    X = model[words]
    result = pca.transform(X)

    plt.scatter(result[:, 0], result[:, 1])
    for i, word in enumerate(words):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))

    plt.show()


def preprocess(lines):
    for i, line in enumerate(lines):
        lines[i] = preprocess_string(line, [lambda x: x.lower(), strip_tags, strip_multiple_whitespaces,
                                            strip_punctuation])


def language_map(lang):
    lang = lang.lower()

    if lang == "dutch" or lang == "nl":
        return ["dutch", "nl"]

    if lang == "russian" or lang == "ru":
        return ["russian", "ru"]

    if lang == "english" or lang == "en":
        return ["english", "en"]

    print("Warning: Language not detected, returning None")
    return None, None


def load_subtitles(lang="nl", size=-1, start=None, end=None):
    lang_full, lang_short = language_map(lang)

    filename = "OpenSubtitles.en"
    if lang_short is not "en":
        filename += "-" + lang_short + "." + lang_short

    file = "subtitle_data/" + lang_full + "/" + filename

    with open(file) as f:
        if start is not None and end is not None:
            subs = []
            line = f.readline()
            ind = 0
            while line:
                if start <= ind <= end:
                    subs.append(line)

                line = f.readline()
                ind += 1
                if ind > end:
                    break

            return subs

        elif size == -1:
            subs = f.read().splitlines()
            return subs
        else:
            subs = []
            for i in range(size):
                sentence = f.readline()
                subs.append(sentence)

            return subs


def get_num_lines(lang):
    lang_full, lang_short = language_map(lang)

    filename = "OpenSubtitles.en"
    if lang_short is not "en":
        filename += "-" + lang_short + "." + lang_short

    file = "subtitle_data/" + lang_full + "/" + filename
    with open(file) as f:
        line = f.readline()
        lines = 0
        while line:
            line = f.readline()
            lines += 1

        return lines