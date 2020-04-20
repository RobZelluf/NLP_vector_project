import random
from utils import language_map, get_num_lines, keep_lines
from time import time
import os
import argparse

random.seed(91)
save_interval = 5e6


def split(lang, train=0.6, val=0.2, test=0.2, filter_lines=False):
    assert train + val + test == 1.0

    lang_full, lang_short = language_map(lang)
    print("Splitting", lang_full.capitalize())
    print("Filtering:", filter_lines)

    if not os.path.exists("data/train_data/"):
        os.mkdir("data/train_data/")

    if not os.path.exists("data/train_data/en_" + lang_short):
        os.mkdir("data/train_data/en_" + lang_short)

    if filter_lines:
        if not os.path.exists("data/train_data/en_" + lang_short + "_filtered"):
            os.mkdir("data/train_data/en_" + lang_short + "_filtered")

    path = "data/subtitle_data/en_" + lang_short + "/OpenSubtitles.en-" + lang_short + ".en"
    num_lines = get_num_lines(path)
    print("Number of lines", num_lines)

    lines = list(range(num_lines))
    train_ind = lines[:int(num_lines * train)]
    val_ind = lines[int(num_lines * train): int(num_lines * (train + val))]
    test_ind = lines[int((train + val) * num_lines):]

    assert len(lines) == len(train_ind) + len(val_ind) + len(test_ind)

    print("Shuffling")
    random.shuffle(lines)
    print("Done shuffling")

    train_lines = [lines[i] for i in train_ind]
    val_lines = [lines[i] for i in val_ind]
    test_lines = [lines[i] for i in test_ind]

    train_lines = sorted(train_lines, reverse=True)
    val_lines = sorted(val_lines, reverse=True)
    test_lines = sorted(test_lines, reverse=True)

    path_en = "data/subtitle_data/en_" + lang_short + "/OpenSubtitles.en-" + lang_short + ".en"
    path_to = "data/subtitle_data/en_" + lang_short + "/OpenSubtitles.en-" + lang_short + "." + lang_short

    # Make training data
    saved_lines = 0
    print("Start reading file")
    with open(path_en, "r") as file1, open(path_to) as file2:
        line_num = 0
        line1 = file1.readline()
        line2 = file2.readline()

        train_lines1 = []
        train_lines2 = []

        val_lines1 = []
        val_lines2 = []

        test_lines1 = []
        test_lines2 = []

        next_train_line = train_lines.pop()
        next_val_line = val_lines.pop()
        next_test_line = test_lines.pop()

        while line1 and line2:
            if (line_num + 1) % save_interval == 0:
                print(lang_full.capitalize() + " - Read", line_num + 1, "out of", num_lines, "lines.")

            save = True
            if filter_lines:
                if not keep_lines(line1, line2):
                    save = False

            if line_num == next_train_line:
                if save:
                    train_lines1.append(line1)
                    train_lines2.append(line2)
                    saved_lines += 1

                if train_lines:
                    next_train_line = train_lines.pop()

            if line_num == next_val_line:
                if save:
                    val_lines1.append(line1)
                    val_lines2.append(line2)
                    saved_lines += 1

                if val_lines:
                    next_val_line = val_lines.pop()

            if line_num == next_test_line:
                if save:
                    test_lines1.append(line1)
                    test_lines2.append(line2)
                    saved_lines += 1

                if test_lines:
                    next_test_line = test_lines.pop()

            if len(train_lines1) >= save_interval:
                save_lines("train", lang_short, train_lines1, train_lines2, filter_lines)
                train_lines1 = []
                train_lines2 = []

            if len(val_lines1) >= save_interval:
                save_lines("val", lang_short, val_lines1, val_lines2, filter_lines)
                val_lines1 = []
                val_lines2 = []

            if len(test_lines1) >= save_interval:
                save_lines("test", lang_short, test_lines1, test_lines2, filter_lines)

                test_lines1 = []
                test_lines2 = []

            line_num += 1
            line1 = file1.readline()
            line2 = file2.readline()

        if train_lines1:
            save_lines("train", lang_short, train_lines1, train_lines2, filter_lines)

        if val_lines1:
            save_lines("val", lang_short, train_lines1, train_lines2, filter_lines)

        if test_lines1:
            save_lines("test", lang_short, train_lines1, train_lines2, filter_lines)

        print("Saved", saved_lines / num_lines * 100, "% of the lines")


def save_lines(set_type, lang_short, lines1, lines2, filter=False):
    print("Saving " + set_type + " files. Filtered:", filter)

    if filter:
        dir1 = "data/train_data/" + "en_" + lang_short + "_filtered/en_" + set_type + ".txt"
        dir2 = "data/train_data/" + "en_" + lang_short + "_filtered/" + lang_short + "_" + set_type + ".txt"
    else:
        dir1 = "data/train_data/" + "en_" + lang_short + "/en_" + set_type + ".txt"
        dir2 = "data/train_data/" + "en_" + lang_short + "/" + lang_short + "_" + set_type + ".txt"

    with open(dir1, "a") as new_file1:
        new_file1.writelines(lines1)

    with open(dir2, "a") as new_file2:
        new_file2.writelines(lines2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", action='store_true')
    args = parser.parse_args()

    if args.filter:
        print("Filtering subtitles")

    for lang in ["dutch", "russian"]:

        t = time()
        split(lang, filter_lines=args.filter)
        print('Splitting up everything took {} mins'.format(round((time() - t) / 60, 2)))