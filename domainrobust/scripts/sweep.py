# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright 2023 Google LLC

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.


import argparse
import copy
import hashlib
import json
import os
import shutil

import numpy as np

from domainrobust import datasets
from domainrobust import algorithms
from domainrobust.lib import misc
from domainrobust import command_launchers

import tqdm
import shlex

class Job:
    NOT_LAUNCHED = 'Not launched'
    INCOMPLETE = 'Incomplete'
    DONE = 'Done'

    def __init__(self, train_args, sweep_output_dir):
        args_str = json.dumps(train_args, sort_keys=True)
        args_hash = hashlib.md5(args_str.encode('utf-8')).hexdigest()
        self.output_dir = os.path.join(sweep_output_dir, args_hash)

        self.train_args = copy.deepcopy(train_args)
        self.train_args['output_dir'] = self.output_dir
        command = ['python3', '-m', 'domainrobust.scripts.train']
        for k, v in sorted(self.train_args.items()):
            if isinstance(v, list):
                v = ' '.join([str(v_) for v_ in v])
            elif isinstance(v, str):
                v = shlex.quote(v)
            command.append(f'--{k} {v}')
        self.command_str = ' '.join(command)
        if os.path.exists(os.path.join(self.output_dir, 'done')):
            self.state = Job.DONE
        elif os.path.exists(self.output_dir):
            self.state = Job.INCOMPLETE
        else:
            self.state = Job.NOT_LAUNCHED

    def __str__(self):
        job_info = (self.train_args['dataset'],
            self.train_args['algorithm'],
            self.train_args['source_envs'],
            self.train_args['target_envs'],
            self.train_args['hparams_seed'])
        return '{}: {} {}'.format(
            self.state,
            self.output_dir,
            job_info)

    @staticmethod
    def launch(jobs, launcher_fn):
        print('Launching...')
        jobs = jobs.copy()
        np.random.shuffle(jobs)
        print('Making job directories:')
        for job in tqdm.tqdm(jobs, leave=False):
            os.makedirs(job.output_dir, exist_ok=True)
        commands = [job.command_str for job in jobs]
        launcher_fn(commands)
        print(f'Launched {len(jobs)} jobs!')

    @staticmethod
    def delete(jobs):
        print('Deleting...')
        for job in jobs:
            shutil.rmtree(job.output_dir)
        print(f'Deleted {len(jobs)} jobs!')

def all_test_env_combinations(n):
    """
    For a dataset with n >= 3 envs, return all combinations of 1 and 2 test
    envs.
    """
    assert(n >= 3)
    for i in range(n):
        yield [i]
        for j in range(i+1, n):
            yield [i, j]


def make_args_list(
    n_trials,
    dataset_names,
    algorithms,
    n_hparams_from,
    n_hparams,
    steps,
    data_dir,
    task,
    holdout_fraction,
    uda_holdout_fraction,
    source_envs,
    target_envs,
    attack,
    source_type,
    pretrain,
    pretrain_model_dir,
    eps,
    atk_iter,
    atk_lr,
    hparams,
):
  args_list = []
  if task == 'domain_adaptation':
    for trial_seed in range(n_trials):
      for dataset in dataset_names:
        for algorithm in algorithms:
          for source_env in source_envs:
            for target_env in target_envs:
              for hparams_seed in range(n_hparams_from, n_hparams):
                train_args = {}
                train_args['dataset'] = dataset
                train_args['algorithm'] = algorithm
                train_args['source_envs'] = source_env
                train_args['target_envs'] = target_env
                train_args['holdout_fraction'] = holdout_fraction
                train_args['uda_holdout_fraction'] = uda_holdout_fraction
                train_args['hparams_seed'] = hparams_seed
                train_args['data_dir'] = data_dir
                train_args['task'] = task
                train_args['trial_seed'] = trial_seed
                train_args['seed'] = misc.seed_hash(
                    dataset, algorithm, target_envs, hparams_seed, trial_seed
                )
                if steps is not None:
                    train_args['steps'] = steps
                if hparams is not None:
                    train_args['hparams'] = hparams
                train_args['attack'] = attack
                train_args['source_type'] = source_type
                if pretrain:
                  train_args['pretrain'] = 1
                else:
                  train_args['pretrain'] = 0
                train_args['pretrain_model_dir'] = pretrain_model_dir
                train_args['eps'] = eps
                train_args['atk_iter'] = atk_iter
                train_args['atk_lr'] = atk_lr
                args_list.append(train_args)
  else:
    raise NotImplementedError
  return args_list


def ask_for_confirmation():
  response = input('Are you sure? (y/n) ')
  if not response.lower().strip()[:1] == 'y':
    print('Nevermind!')
    exit(0)


DATASETS = [d for d in datasets.DATASETS if 'Debug' not in d]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a sweep')
    parser.add_argument('command', choices=['launch', 'delete_incomplete'])
    parser.add_argument('--datasets', nargs='+', type=str, default=DATASETS)
    parser.add_argument(
        '--algorithms', nargs='+', type=str, default=algorithms.ALGORITHMS
    )
    parser.add_argument('--task', type=str, default='domain_adaptation')
    parser.add_argument('--n_hparams_from', type=int, default=0)
    parser.add_argument('--n_hparams', type=int, default=20)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_trials', type=int, default=3)
    parser.add_argument('--command_launcher', type=str, required=True)
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--hparams', type=str, default=None)
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0.75)
    parser.add_argument('--source_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--target_envs', type=int, nargs='+', default=[1])
    parser.add_argument('--eps', type=float, default=0.0)
    parser.add_argument('--atk_iter', type=int, default=20)
    parser.add_argument('--atk_lr', type=float, default=0.004)
    parser.add_argument('--attack', type=str, default='None')
    parser.add_argument('--source_type', type=str, default='clean')
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--pretrain_model_dir', type=str, default=None,
        help='pretrain model weight directory')
    parser.add_argument('--skip_confirmation', action='store_true')
    args = parser.parse_args()
    args_list = make_args_list(
        n_trials=args.n_trials,
        dataset_names=args.datasets,
        algorithms=args.algorithms,
        n_hparams_from=args.n_hparams_from,
        n_hparams=args.n_hparams,
        steps=args.steps,
        data_dir=args.data_dir,
        task=args.task,
        holdout_fraction=args.holdout_fraction,
        uda_holdout_fraction=args.uda_holdout_fraction,
        source_envs=args.source_envs,
        target_envs=args.target_envs,
        attack=args.attack,
        source_type=args.source_type,
        pretrain=args.pretrain,
        pretrain_model_dir = args.pretrain_model_dir,
        eps=args.eps,
        atk_iter=args.atk_iter,
        atk_lr=args.atk_lr,
        hparams=args.hparams,
    )

    jobs = [Job(train_args, args.output_dir) for train_args in args_list]

    for job in jobs:
        print(job)
    print("{} jobs: {} done, {} incomplete, {} not launched.".format(
        len(jobs),
        len([j for j in jobs if j.state == Job.DONE]),
        len([j for j in jobs if j.state == Job.INCOMPLETE]),
        len([j for j in jobs if j.state == Job.NOT_LAUNCHED]))
    )

    if args.command == 'launch':
        to_launch = [j for j in jobs if j.state == Job.NOT_LAUNCHED]
        print(f'About to launch {len(to_launch)} jobs.')
        if not args.skip_confirmation:
            ask_for_confirmation()
        launcher_fn = command_launchers.REGISTRY[args.command_launcher]
        Job.launch(to_launch, launcher_fn)

    elif args.command == 'delete_incomplete':
        to_delete = [j for j in jobs if j.state == Job.INCOMPLETE]
        print(f'About to delete {len(to_delete)} jobs.')
        if not args.skip_confirmation:
            ask_for_confirmation()
        Job.delete(to_delete)
