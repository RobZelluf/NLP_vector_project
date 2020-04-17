from gensim.models import Word2Vec
from utilities.utils import language_map, preprocess, load_subtitles,\
    get_num_lines, add_special_tokens, load_random_subtitles
import logging
import argparse
import random
import math
import multiprocessing
import os


def train_chunk(model, language, p, epochs, special_tokens):
    subtitles = load_random_subtitles(language, p)
    print("Training on", len(subtitles), "lines")
    preprocess(subtitles, False)
    if special_tokens:
        add_special_tokens(subtitles)

    model.build_vocab(subtitles, progress_per=10000, update=model.wv.vocab)
    model.train(subtitles, total_examples=len(subtitles), epochs=epochs)


def train(args):
    lang_full, lang_short = language_map(args.language)

    model_name = lang_short + "_d" + str(args.dim) + ".model"
    if args.special_tokens:
        model_name = lang_short + "_d" + str(args.dim) + "_st.model"

    model_path = "trained_models/" + lang_full + "/" + model_name

    print("Training on", lang_full, "- Embedding size:", args.dim, "- Loops:", args.loops,
          "- Chunks:", args.chunks, "- Epochs:", args.epochs)

    cores = max(1, multiprocessing.cpu_count() - 2)
    model = Word2Vec(min_count=20,
                     window=5,
                     size=args.dim,
                     alpha=0.03,
                     min_alpha=0.0007,
                     workers=cores,
                     sample=6e-5,
                     negative=20)

    if os.path.exists(model_path) and args.continue_training:
        print("Continuing training existing model")
        model = Word2Vec.load(model_path)

    model.workers = cores

    num_lines = get_num_lines(args.language)
    print("Total number of lines:", num_lines)
    chunk_size = int(num_lines / args.chunks)
    if chunk_size > 5e7:
        chunk_size = 5e7
        args.chunks = int(math.ceil(num_lines / chunk_size))
        print("Chunk size too large, set to", int(chunk_size), "with", args.chunks, "chunks!")

    p = chunk_size / num_lines

    for loop in range(args.loops):
        chunk_list = list(range(args.chunks))
        random.shuffle(chunk_list)
        for i, chunk in enumerate(chunk_list):
            print("Loop", loop + 1, "/", args.loops, "- Chunk, ", i + 1, "/", args.chunks)

            train_chunk(model, args.language, p, args.epochs, args.special_tokens)
            model.save(model_path)
            print("Model saved as", model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, help="Language to train", default="en")
    parser.add_argument("--loops", type=int, help="Number of full loops over the corpus", default=1)
    parser.add_argument("--chunks", type=int, help="Number of chunks to split corpus in", default=10)
    parser.add_argument("--epochs", type=int, help="Number of epochs per chunk", default=5)
    parser.add_argument("--dim", type=int, help="Embedding dimension", default=300)
    parser.add_argument("--log", type=bool, help="Pass if you want gensim to print logs")
    parser.add_argument("--continue_training", type=bool, default=True)
    parser.add_argument("--special_tokens", action='store_true')
    args = parser.parse_args()

    if args.log:
        print("Gensim will output logs!")
        logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

    if not os.path.exists("trained_models"):
        os.mkdir("trained_models")

    lang, _ = language_map(args.language)

    if not os.path.exists("trained_models/" + lang):
        os.mkdir("trained_models/" + lang)

    if args.special_tokens:
        print("Adding tokens <SOS> and <EOS>")

    train(args)





