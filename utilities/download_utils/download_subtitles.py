import csv
import os
import requests
from zipfile import ZipFile
import gzip


def download_url(url, DIR):
    filename = url.split("/")[-1]
    save_path = DIR + "/" + filename

    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)


def download_subtitle_files(datasets):
    print("\nDownloading languages\n")
    for language, url in datasets:
        print("Downloading", language)
        DIR = "subtitle_data/" + language
        if not os.path.exists(DIR):
            os.mkdir(DIR)

        else:
            if os.listdir(DIR):
                print(language, "already downloaded")
                continue

        download_url(url, DIR)
        print("Done downloading", language)


def extract_files():
    print("\nUnzipping files\n")
    for language_dir in os.listdir("subtitle_data"):
        if os.path.isdir("subtitle_data/" + language_dir):
            if language_dir == "english":

                for file in os.listdir("subtitle_data/" + language_dir):
                    ext = os.path.splitext(file)
                    if ext[-1] == ".gz":
                        DIR = "subtitle_data/" + language_dir
                        new_filename = ext[0]
                        with gzip.GzipFile(DIR + "/" + file, 'rb') as gzip_file:
                            with open(DIR + "/" + new_filename, 'wb') as new_file:
                                for data in iter(lambda: gzip_file.read(1024 * 1024), b''):
                                    new_file.write(data)

                        os.remove(DIR + "/" + file)
            else:
                for file in os.listdir("subtitle_data/" + language_dir):
                    ext = os.path.splitext(file)
                    if ext[-1] == '.zip':
                        print("Unzipping", language_dir)
                        DIR = "subtitle_data/" + language_dir + "/"
                        with ZipFile(DIR + file, 'r') as f:
                            f.extractall(DIR)

                        os.remove(DIR + file)
                        print("Done unzipping", file)


def download_all_subtitles():
    if not os.path.exists("subtitle_data"):
        os.mkdir("subtitle_data")

    with open("utilities/download_utils/subtitle_download_data.csv", "r") as f:
        datasets = csv.reader(f)
        datasets = list(datasets)

    download_subtitle_files(datasets)
    extract_files()


download_all_subtitles()