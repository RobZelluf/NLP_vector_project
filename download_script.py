from utilities.download_utils.download_subtitles import download_all_subtitles
from utilities.download_utils.download_vector_models import download_all_vector_models
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_subtitles", action="store_true")
    parser.add_argument("--skip_embeddings", action="store_true")

    args = parser.parse_args()

    if not args.skip_subtitles:
        print("Downloading all subtitles")
        download_all_subtitles()

    if not args.skip_embeddings:
        print("Downloading all vector models")
        download_all_vector_models()