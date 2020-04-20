import os
import random
import matplotlib.pyplot as plt
from utilities.utils import get_num_lines
import numpy as np

lang_short = "nl"

path_en = "data/subtitle_data/en_" + lang_short + "/OpenSubtitles.en-" + lang_short + ".en"
path_to = "data/subtitle_data/en_" + lang_short + "/OpenSubtitles.en-" + lang_short + "." + lang_short
path_ids = "data/subtitle_data/en_" + lang_short + "/OpenSubtitles.en-" + lang_short + ".ids"


def sample_lines(num_lines, threshold):
    kept = 0
    discarded = 0

    with open(path_en, "r") as file1, open(path_to) as file2:
        line1 = file1.readline()
        line2 = file2.readline()
        while line1 and line2 and (kept < num_lines or discarded < num_lines):
            if random.random() < 0.0001:
                max_length = max(len(line1), len(line2))
                threshold_length = max(15, threshold * max_length)

                diff = abs(len(line1) - len(line2))
                if diff < threshold_length:
                    if kept < num_lines and diff > 5:
                        kept += 1
                        print("\nKEEPING")
                        print("Threshold length", threshold_length)
                        print("Difference:", diff)
                        print(line1)
                        print(line2)
                elif discarded < num_lines:
                    discarded += 1
                    print("\nDISCARDING")
                    print("Threshold length", threshold_length)
                    print("Difference:", diff)
                    print(line1)
                    print(line2)

            line1 = file1.readline()
            line2 = file2.readline()


def line_shrinkage(threshold):
    num_lines = get_num_lines(path_en)

    original_lengths = 0
    new_lengths = 0

    lines_keps = 0
    counter = 0
    with open(path_en, "r") as file1, open(path_to) as file2:
        line1 = file1.readline()
        line2 = file2.readline()
        while line1 and line2:
            counter += 1
            if counter % 1e6 == 0:
                print("Read", counter, "out of", num_lines)

            diff = abs(len(line1) - len(line2))
            apr_length = (len(line1) + len(line2)) / 2
            if diff < threshold:
                lines_keps += 1
                new_lengths += apr_length

            original_lengths += apr_length

            line1 = file1.readline()
            line2 = file2.readline()

    new_lengths /= lines_keps
    original_lengths /= num_lines

    return new_lengths / original_lengths


def get_percentage(thresholds):
    num_lines = get_num_lines(path_en)

    counts = np.zeros(len(thresholds))
    counter = 0
    with open(path_en, "r") as file1, open(path_to) as file2:
        line1 = file1.readline()
        line2 = file2.readline()
        while line1 and line2:
            counter += 1
            if counter % 1e6 == 0:
                print("Read", counter, "out of", num_lines)

            diff = abs(len(line1) - len(line2))
            for i, threshold in enumerate(thresholds):
                if diff < threshold:
                    counts[i] += 1
                else:
                    break

            line1 = file1.readline()
            line2 = file2.readline()

    return counts / num_lines


def plot_cutoff():
    threshold_range = (list(range(10, 40)))
    threshold_range.reverse()

    percentages = get_percentage(threshold_range)

    plt.plot(threshold_range, percentages)
    plt.xlabel("Cut-off threshold in number of characters")
    plt.ylabel("Percentage of corpus kept")
    plt.show()


sample_lines(20, 0.35)
