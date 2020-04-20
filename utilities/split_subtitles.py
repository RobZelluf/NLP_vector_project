import random
from utils import language_map, get_num_lines
from time import time

random.seed(91)
save_interval = 1e6


def split(lang, train=0.6, val=0.2, test=0.2):
    assert train + val + test == 1.0

    lang_full, lang_short = language_map(lang)
    print("Splitting", lang_full.capitalize())
    num_lines = get_num_lines(lang)

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

    path_en = "data/subtitle_data/" + lang_full + "/OpenSubtitles.en-" + lang_short + ".en"
    path_to = "data/subtitle_data/" + lang_full + "/OpenSubtitles.en-" + lang_short + "." + lang_short

    # Make training data
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
                print(lang_full.capitalize() + "- Read", line_num + 1, "out of", num_lines, "lines.")

            if line_num == next_train_line:
                train_lines1.append(line1)
                train_lines2.append(line2)

                if train_lines:
                    next_train_line = train_lines.pop()

            if line_num == next_val_line:
                val_lines1.append(line1)
                val_lines2.append(line2)

                if val_lines:
                    next_val_line = val_lines.pop()

            if line_num == next_test_line:
                test_lines1.append(line1)
                test_lines2.append(line2)

                if test_lines:
                    next_test_line = test_lines.pop()

            line_num += 1
            line1 = file1.readline()
            line2 = file2.readline()

            if len(train_lines1) >= save_interval:
                save_train_lines("train", lang_short, train_lines1, train_lines2)
                train_lines1 = []
                train_lines2 = []

            if len(val_lines1) >= save_interval:
                save_train_lines("val", lang_short, val_lines1, val_lines2)
                val_lines1 = []
                val_lines2 = []

            if len(test_lines1) >= save_interval:
                save_train_lines("test", lang_short, test_lines1, test_lines2)

                test_lines1 = []
                test_lines2 = []


def save_train_lines(set_type, lang_short, lines1, lines2):
    print("Saving " + set_type + " files")
    with open("data/train_data/" + "en_" + lang_short + "/en_" + set_type + ".txt", "a") as new_file1:
        new_file1.writelines(lines1)

    with open("data/train_data/" + "en_" + lang_short + "/" + lang_short + "_" + set_type + ".txt", "a") as new_file2:
        new_file2.writelines(lines2)


if __name__ == "__main__":
    for lang in ["dutch", "russian"]:
        t = time()
        split(lang)
        print('Splitting up everything took {} mins'.format(round((time() - t) / 60, 2)))