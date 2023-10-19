# Copyright 2023 Google LLC

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Evaluation."""

import argparse
import json
import os
import random
import sys

from domainrobust import algorithms
from domainrobust import datasets
from domainrobust.lib import misc
from domainrobust.lib.fast_data_loader import FastDataLoader
import numpy as np
import torch

if __name__ == '__main__':
  np.set_printoptions(suppress=True)

  parser = argparse.ArgumentParser(description='Domain adaptation testbed')
  parser.add_argument('--input_dir', type=str, required=True)
  parser.add_argument('--data_dir', type=str)
  parser.add_argument('--dataset', type=str, default='DIGIT')
  parser.add_argument('--attack', type=str, default='None, pgd, aa')
  parser.add_argument('--eps', type=float, default=0.0)
  parser.add_argument('--atk_iter', type=int, default=10)
  parser.add_argument('--atk_lr', type=float, default=0.004)
  parser.add_argument('--restart', type=int, default=1)
  args = parser.parse_args()
  oracle_acc, oracle_clean_acc, oracle_val_acc = [], [], []
  paths = []
  epochs_path = ""
  with open(
      os.path.join(args.input_dir, 'best_model_path_pgd.txt'), 'r'
  ) as f:
    for line in f:
      paths.append(line[:-1])
  for subdir in paths:
    subdir = os.path.basename(os.path.normpath(subdir))
    results_path = os.path.join(args.input_dir, subdir, 'results.jsonl')
    records = None
    with open(results_path, 'r') as f:
      for line in f:
        records = json.loads(line[:-1])
        break
    if records is None:
      raise EOFError

    seed = records['args']['seed']
    hparams_seed = records['args']['hparams_seed']
    hparams = records['hparams']
    holdout_fraction = records['args']['holdout_fraction']
    trial_seed = records['args']['trial_seed']
    uda_holdout_fraction = records['args']['uda_holdout_fraction']
    source_envs = records['args']['source_envs']
    target_envs = records['args']['target_envs']
    task = records['args']['task']
    algorithm_name = records['args']['algorithm']

    if algorithm_name in ['ERM', 'DANN']:
      epochs_path = os.path.join(args.input_dir, 'results_clean.txt')
      model_train_ckpt = os.path.join(
          args.input_dir, subdir, 'model_train_best.pkl'
      )
      model_test_ckpt = os.path.join(
          args.input_dir, subdir, 'model_test_best.pkl'
      )

    else:
      epochs_path = os.path.join(
          args.input_dir,
          'results_pgd_eps'
          + str(args.eps)
          + '_step'
          + str(args.atk_iter)
          + '_restart'
          + str(args.restart)
          + '.txt',
      )
      model_train_ckpt = os.path.join(
          args.input_dir, subdir, 'model_train_pgd_best.pkl'
      )
      model_test_ckpt = os.path.join(
          args.input_dir, subdir, 'model_test_pgd_best.pkl'
      )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
      device = 'cuda'
    else:
      device = 'cpu'

    if args.dataset in vars(datasets):
      dataset = vars(datasets)[args.dataset](args.data_dir, target_envs, hparams)
    else:
      raise NotImplementedError

    in_splits = []
    out_splits = []
    uda_splits = []
    for env_i, env in enumerate(dataset):
      uda = []
      if task == 'domain_adaptation':
        in_, out, uda = None, None, None
        if env_i in source_envs:
          out, in_ = misc.split_dataset(
              env,
              int(len(env) * holdout_fraction),
              misc.seed_hash(trial_seed, env_i),
          )
        elif env_i in target_envs:
          out, in_ = misc.split_dataset(
              env,
              int(len(env) * holdout_fraction),
              misc.seed_hash(trial_seed, env_i),
          )
          uda, in_ = misc.split_dataset(
              in_,
              int(len(in_) * uda_holdout_fraction),
              misc.seed_hash(trial_seed, env_i),
          )
        if records['hparams']['class_balanced'] and in_ is not None:
          in_weights = misc.make_weights_for_balanced_classes(in_)
          out_weights = misc.make_weights_for_balanced_classes(out)
          if uda is not None:
            uda_weights = misc.make_weights_for_balanced_classes(uda)
          else:
            uda_weights = None
        else:
          in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        uda_splits.append((uda, uda_weights))
      else:
        raise NotImplementedError
    eval_loaders = [
        FastDataLoader(
            dataset=env, batch_size=64, num_workers=dataset.N_WORKERS)
        for env, _ in (in_splits + out_splits + uda_splits)
        if env is not None]
    eval_weights = [
        None for _, _ in (in_splits + out_splits + uda_splits)
    ]
    eval_loader_names = [
        'env{}_in'.format(i)
        for i in range(len(in_splits))
        if in_splits[i][0] is not None
    ]
    eval_loader_names += [
        'env{}_out'.format(i)
        for i in range(len(out_splits))
        if out_splits[i][0] is not None
    ]
    eval_loader_names += [
        'env{}_uda'.format(i)
        for i in range(len(uda_splits))
        if uda_splits[i][0] is not None
    ]

    algorithm_class = algorithms.get_algorithm_class(algorithm_name)
    if task == 'domain_adaptation':
      num_domain = 2
    else:
      raise NotImplementedError
    algorithm = algorithm_class(
        dataset.input_shape,
        dataset.num_classes,
        num_domain,
        hparams,
    )
    evals = zip(eval_loader_names, eval_loaders, eval_weights)
    results = {}
    for name, loader, _ in evals:
      # eval on adversarial examples
      if name == 'env{}_in'.format(target_envs[0]):
        algorithm.load_state_dict(torch.load(model_test_ckpt)['model_dict'])
        algorithm.to(device)
        acc = misc.accuracy(algorithm, loader, device)
        oracle_clean_acc.append(acc)
        if args.attack == 'pgd':
          pgd_acc = misc.pgd_accuracy(
              algorithm,
              loader,
              args.eps,
              args.atk_lr,
              args.atk_iter,
              device,
              args.restart,
          )
          results[name + '_pgd_acc_oracle'] = pgd_acc
          oracle_acc.append(pgd_acc)
        elif args.attack == 'aa':
          aa_acc = misc.autoattack_accuracy(
              algorithm_name,
              algorithm,
              loader,
              args.eps,
              device,
          )
          oracle_acc.append(aa_acc)
        elif args.attack == 'None':
          pass
        else:
          raise NotImplementedError

  sys.stdout = misc.Tee(epochs_path, 'w')
  mean = 100 * np.mean(list(oracle_acc))
  err = 100 * np.std(list(oracle_acc) / np.sqrt(len(oracle_acc)))
  print('target robust accuracy:{:.1f} +/- {:.1f}'.format(mean, err))
  print(oracle_acc)
  mean = 100 * np.mean(list(oracle_clean_acc))
  err = 100 * np.std(list(oracle_clean_acc) / np.sqrt(len(oracle_clean_acc)))
  print('target clean accuracy:{:.1f} +/- {:.1f}'.format(mean, err))
  print(oracle_clean_acc)
