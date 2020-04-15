from gensim.models import Word2Vec
from utilities.utils import language_map, preprocess, load_subtitles
import logging
import argparse
import random
import math
import multiprocessing
import os


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


def train_chunk(model, language, epochs, start, end):
    subtitles = load_subtitles(language, start=start, end=end)
    preprocess(subtitles)
    print("Updating vocabulary")

    model.build_vocab(subtitles, progress_per=10000, update=model.wv.vocab)
    print("Training model")
    model.train(subtitles, total_examples=len(subtitles), epochs=epochs)


def train(language, dim=100, loops=1, epochs=10, chunks=10, continue_training=True):
    lang_full, lang_short = language_map(language)
    model_name = lang_short + "_d" + str(dim) + ".model"
    model_path = "vector_models/" + lang_full + "/" + model_name

    print("Training on", lang_full, "- Embedding size:", dim, "- Loops:", loops, "- Chunks:", chunks)

    cores = max(1, multiprocessing.cpu_count() - 2)
    model = Word2Vec(min_count=20,
                     window=5,
                     size=300,
                     alpha=0.03,
                     min_alpha=0.0007,
                     workers=cores,
                     sample=6e-5,
                     negative=20)

    if os.path.exists(model_path) and continue_training:
        print("Continuing training existing model")
        model = Word2Vec.load(model_path)

    model.workers = cores

    num_lines = get_num_lines(language)
    chunk_size = int(num_lines / chunks)
    if chunk_size > 5e6:
        chunk_size = 5e6
        chunks = int(math.ceil(num_lines / chunk_size))
        print("Chunk size too large, set to", int(chunk_size), "with", chunks, "chunks!")

    for loop in range(loops):
        chunk_list = list(range(chunks))
        random.shuffle(chunk_list)
        for i, chunk in enumerate(chunk_list):
            print("Loop", loop, "/", loops, "- Chunk, ", i + 1, "/", chunks)
            start = chunk * chunk_size
            if chunk == chunks - 1:
                end = num_lines - 1
            else:
                end = (chunk + 1) * chunk_size - 1

            train_chunk(model, language, epochs, start, end)
            model.save(model_path)
            print("Model saved as", model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, help="Language to train", default="en")
    parser.add_argument("--loops", type=int, help="Number of full loops over the corpus", default=1)
    parser.add_argument("--chunks", type=int, help="Number of chunks to split corpus in", default=10)
    parser.add_argument("--epochs", type=int, help="Number of epochs per chunk", default=10)
    parser.add_argument("--dim", type=int, help="Embedding dimension", default=300)
    parser.add_argument("--log", type=bool, help="Pass if you want gensim to print logs")
    parser.add_argument("--continue_training", type=bool, default=True)
    args = parser.parse_args()

    if args.log:
        print("Gensim will output logs!")
        logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

    if not os.path.exists("vector_models"):
        os.mkdir("vector_models")

    lang, _ = language_map(args.language)

    if not os.path.exists("vector_models/" + lang):
        os.mkdir("vector_models/" + lang)

    train(args.language, args.dim, args.loops, args.epochs, args.chunks, args.continue_training)





