import os
import csv
from utils.download_utils.wikipedia2vec import *
from utils.download_utils.glove import *


def download_all_vector_models():
    with open("utils/download_utils/subtitle_download_data.csv", "r") as f:
        data = csv.reader(f)
        data = list(data)
        languages = [x[0] for x in data]
        languages.append("english")

    if not os.path.exists("vector_models"):
        os.mkdir("vector_models")

    for language in languages:
        if not os.path.exists("vector_models/" +language):
            os.mkdir("vector_models/" + language)

    download_wikipedia2vec(languages, 100)
    extract_wikipedia2vec_files()

    download_glove()
    extract_glove_files()
