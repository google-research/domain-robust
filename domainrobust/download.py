# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright 2023 Google LLC

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.


from zipfile import ZipFile
import argparse
import tarfile
import gdown
import os


# utils #######################################################################

def stage_path(data_dir, name):
    full_path = os.path.join(data_dir, name)

    if not os.path.exists(full_path):
        os.makedirs(full_path)

    return full_path


def download_and_extract(url, dst, remove=True):
    gdown.download(url, dst, quiet=False)

    if dst.endswith(".tar.gz"):
        tar = tarfile.open(dst, "r:gz")
        tar.extractall(os.path.dirname(dst))
        tar.close()

    if dst.endswith(".tar"):
        tar = tarfile.open(dst, "r:")
        tar.extractall(os.path.dirname(dst))
        tar.close()

    if dst.endswith(".zip"):
        zf = ZipFile(dst, "r")
        zf.extractall(os.path.dirname(dst))
        zf.close()

    if remove:
        os.remove(dst)

# DIGIT ########################################################################

def download_digit(data_dir):
    # Original URL: https://github.com/VisionLearningGroup/VisionLearningGroup.github.io/tree/master/M3SDA/code_MSDA_digit
    full_path = stage_path(data_dir, "DIGIT")

    download_and_extract("https://drive.google.com/uc?id=1wnY5M74Zj-lPuOQzcmVO9KQVLVhgK7Vu",
                         os.path.join(data_dir, "DIGIT.zip"))

    os.rename(os.path.join(data_dir, "kfold"),
              full_path)



# PACS ########################################################################

def download_pacs(data_dir):
    # Original URL: http://www.eecs.qmul.ac.uk/~dl307/project_iccv2017
    full_path = stage_path(data_dir, "PACS")

    download_and_extract("https://drive.google.com/uc?id=1JFr8f805nMUelQWWmfnJR3y4_SYoN5Pd",
                         os.path.join(data_dir, "PACS.zip"))

    os.rename(os.path.join(data_dir, "kfold"),
              full_path)


# Office-Home #################################################################

def download_office_home(data_dir):
    # Original URL: http://hemanthdv.org/OfficeHome-Dataset/
    full_path = stage_path(data_dir, "office_home")

    download_and_extract("https://drive.google.com/uc?id=1uY0pj7oFsjMxRwaD3Sxy0jgel0fsYXLC",
                         os.path.join(data_dir, "office_home.zip"))

    os.rename(os.path.join(data_dir, "OfficeHomeDataset_10072016"),
              full_path)


# VISDA ########################################################################

def download_visda(data_dir):
    # Original URL: https://ai.bu.edu/visda-2017/
    full_path = stage_path(data_dir, "VISDA")

    download_and_extract("https://drive.google.com/uc?id=1VrIlU6yrm-XTcwfpRIWuALdTAYulgdYp",
                         os.path.join(data_dir, "VISDA.zip"))

    os.rename(os.path.join(data_dir, "kfold"),
              full_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download datasets')
    parser.add_argument('--data_dir', type=str, required=True)
    args = parser.parse_args()

    # download_digit(args.data_dir)
    # download_pacs(args.data_dir)
    # download_office_home(args.data_dir)
    # download_visda(args.data_dir)
