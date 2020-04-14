import csv
import os
import requests
from zipfile import ZipFile


def download_url(url, DIR):
    save_path = DIR + "/subtitle_data.zip"

    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)


if not os.path.exists("subtitle_data"):
    os.mkdir("subtitle_data")

with open("utils/subtitle_download_data.csv", "r") as f:
    datasets = csv.reader(f)
    datasets = list(datasets)

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

print("\nUnzipping files\n")
for language_dir in os.listdir("subtitle_data"):
    if os.path.isdir("subtitle_data/" + language_dir):
        for file in os.listdir("subtitle_data/" + language_dir):
            ext = os.path.splitext(file)
            if ext[-1] == '.zip':
                print("Unzipping", language_dir)
                DIR = "subtitle_data/" + language_dir + "/"
                with ZipFile(DIR + file, 'r') as f:
                    f.extractall(DIR)

                os.remove(DIR + file)
                print("Done unzipping", file)

print("Done!")