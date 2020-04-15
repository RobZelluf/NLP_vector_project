from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_multiple_whitespaces, strip_punctuation


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
