"""Normalize Saint Gall dataset."""

from glob import glob
import shutil
import os


def partitions(origin, path):
    """Normalize and create 'partitions' folder."""

    if os.path.exists(path.partitions):
        shutil.rmtree(path.partitions)
    os.makedirs(path.partitions)

    origin_dir = os.path.join(origin, "sets")

    def complete_partition_file(set_file, new_set_file):
        with open(set_file) as file:
            with open(new_set_file, "w+") as new_file:
                content = [x.strip() for x in file.readlines()]
                lines = os.path.join(origin, "data", "line_images_normalized")

                for item in content:
                    glob_filter = os.path.join(lines, f"{item}*")
                    paths = [x for x in glob(glob_filter, recursive=True)]

                    for path in paths:
                        basename = os.path.basename(path).split(".")[0]
                        new_file.write(f"{basename.strip()}\n")

    set_file = os.path.join(origin_dir, "train.txt")
    complete_partition_file(set_file, path.train_file)

    set_file = os.path.join(origin_dir, "valid.txt")
    complete_partition_file(set_file, path.validation_file)

    set_file = os.path.join(origin_dir, "test.txt")
    complete_partition_file(set_file, path.test_file)


def ground_truth(origin, path):
    """Normalize and create 'ground_truth' folder (Ground Truth)."""

    if os.path.exists(path.ground_truth):
        shutil.rmtree(path.ground_truth)
    os.makedirs(path.ground_truth)

    origin_dir = os.path.join(origin, "ground_truth")
    set_file = os.path.join(origin_dir, "transcription.txt")

    with open(set_file) as file:
        content = [x.strip() for x in file.readlines()]

        for line in content:
            if (not line or line[0] == "#"):
                continue

            splited = line.strip().split(' ')
            assert len(splited) >= 3

            file_name = splited[0]
            file_text = splited[1].replace("-", "").replace("|", " ")

            new_set_file = os.path.join(path.ground_truth, f"{file_name}.txt")

            with open(new_set_file, "w+") as new_file:
                new_file.write(file_text.strip())


def data(origin, path):
    """Normalize and create 'lines' folder."""

    if os.path.exists(path.data):
        shutil.rmtree(path.data)
    os.makedirs(path.data)

    origin_dir = os.path.join(origin, "data")

    glob_filter = os.path.join(origin_dir, "line_images_normalized", "*.*")
    files = [x for x in glob(glob_filter, recursive=True)]

    for file in files:
        name = os.path.basename(file).split(".")[0]
        new_file = os.path.join(path.data, f"{name}.png")
        shutil.copy(file, new_file)