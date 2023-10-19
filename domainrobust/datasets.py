# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright 2023 Google LLC

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.


import os
import torch
from PIL import ImageFile
from torchvision import transforms

from torch.utils.data import TensorDataset
from torchvision.datasets import ImageFolder

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    # Debug
    "Debug32",
    "Debug224",
    # Small images
    "DIGIT",
    # Big images
    "PACS",
    "OfficeHome",
    "VISDA",
]

def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


class MultipleDomainDataset:
    N_STEPS = 50001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 8            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class MultipleEnvironmentDigit(MultipleDomainDataset):

  def __init__(self, root):
    super().__init__()
    environments = [f.name for f in os.scandir(root) if f.is_dir()]
    environments = sorted(environments)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    self.datasets = []
    for _, environment in enumerate(environments):
      env_transform = transform
      path = os.path.join(root, environment)
      env_dataset = ImageFolder(path, transform=env_transform)
      self.datasets.append(env_dataset)

    self.input_shape = (3, 32, 32,)
    self.num_classes = 10



class DIGIT(MultipleEnvironmentDigit):
    N_STEPS = 25001
    ENVIRONMENTS = ["mnist", "mnist-m", "svhn", "syn", "usps"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "DIGIT/")
        super().__init__(self.dir)


class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)

        self.datasets = []
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        for _, environment in enumerate(environments):
            env_transform = transform
            path = os.path.join(root, environment)
            env_dataset = ImageFolder(path, transform=env_transform)
            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = len(self.datasets[-1].classes)


class VISDA(MultipleEnvironmentImageFolder):
    N_STEPS = 25001
    ENVIRONMENTS = ["test", "train", "validation"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "VISDA/")
        super().__init__(self.dir)

class PACS(MultipleEnvironmentImageFolder):
    N_STEPS = 25001
    ENVIRONMENTS = ["A", "C", "P", "S"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "PACS/")
        super().__init__(self.dir)

class OfficeHome(MultipleEnvironmentImageFolder):
    N_STEPS = 25001
    ENVIRONMENTS = ["A", "C", "P", "R"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "office_home/")
        super().__init__(self.dir)