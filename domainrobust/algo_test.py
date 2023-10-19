# Copyright 2023 Google LLC

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import itertools
from absl.testing import absltest
from domainrobust import algorithms
from domainrobust import datasets
from domainrobust import hparams_registry
from parameterized import parameterized
import torch
import os

def make_minibatches(dataset, batch_size):
  """Test helper to make a minibatches array like train.py"""
  minibatches = []
  X = []
  device = "cuda" if torch.cuda.is_available() else "cpu"
  for env in dataset:
    X = torch.stack([env[i][0] for i in range(batch_size)]).to(device)
    y = torch.stack(
        [torch.as_tensor(env[i][1]) for i in range(batch_size)]
    ).to(device)
    minibatches.append((X, y))
  return minibatches, X

class Test(absltest.TestCase):
  @parameterized.expand(itertools.product(datasets.DATASETS))
  @absltest.skipIf(
      'DATA_DIR' not in os.environ, 'needs DATA_DIR environment variable'
  )
  def test_dataset_erm(self, dataset_name):
    """Test that ERM can complete one step on a given dataset without raising
    an error.
    Also test that num_environments() works correctly.
    """
    batch_size = 8
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for algorithm_name in algorithms.ALGORITHMS:
      hparams = hparams_registry.default_hparams(algorithm_name, dataset_name)
      dataset = datasets.get_dataset_class(dataset_name)(
          os.environ['DATA_DIR'], [], hparams
      )
      self.assertEqual(datasets.num_environments(dataset_name), len(dataset))
      algorithm = algorithms.get_algorithm_class(algorithm_name)(
          dataset.input_shape, dataset.num_classes, len(dataset), hparams
      ).to(device)
      minibatches, uda_batch = make_minibatches(dataset, batch_size)
      algorithm.update(minibatches, uda_batch)

if __name__ == "__main__":
  absltest.main()

