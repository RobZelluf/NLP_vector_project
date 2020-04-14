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
