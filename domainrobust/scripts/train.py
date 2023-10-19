# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright 2023 Google LLC

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.


import argparse
import collections
import json
import os
import random
import sys
import time

from torchvision import transforms
import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data

from domainrobust import datasets
from domainrobust import hparams_registry
from domainrobust import algorithms
from domainrobust.lib import misc
from domainrobust.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader

_DIGIT_IMG_SIZE = 32
_IMG_SIZE = 224
# Probability of whether to include the additional regularizer for ROBDANN.

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--task', type=str, default="domain_adaptation",
        choices=["domain_adaptation"])
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--source_envs', type=int, nargs='+', default=[0],
        help='source domain indice')
    parser.add_argument('--target_envs', type=int, nargs='+', default=[0],
        help='target domain indice')
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0.75,
        help='For domain adaptation, % of test to use unlabeled for training.',)
    parser.add_argument('--attack', type=str, default='None', 
        help='Attack type choice: [None, pgd, aa]. None: No attack, pgd: L_infty PGD attack')
    parser.add_argument('--source_type', type=str, default='clean',
                        choices=['clean', 'adv', 'kl'])
    parser.add_argument('--pretrain', type=int, default=0)
    parser.add_argument('--pretrain_model_dir', type=str, default=None,
        help='pretrain model weight directory')
    parser.add_argument('--eps', type=float, default=0.0)
    parser.add_argument('--atk_iter', type=int, default=20)
    parser.add_argument('--atk_lr', type=float, default=0.004)
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    args = parser.parse_args()

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    algorithm_dict = None

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))
    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
            args.target_envs, hparams)
    else:
        raise NotImplementedError

    if args.source_envs == args.target_envs:
        raise ValueError("Source and target domain cannot be the same.")

    # Set the data augmentation.
    if args.dataset in ['DIGIT']:
        img_size = _DIGIT_IMG_SIZE
    else:
        img_size = _IMG_SIZE
    if hparams['data_augmentation']:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    # Split train env into an 'in-split' and an 'out-split'. The 'in-split' is
    # used for training data, the 'out-split' is used for validation data.

    # To allow unsupervised domain adaptation experiments, we split each test
    # env into 'in-split', 'uda-split' and 'out-split'. The 'in-split' is used
    # for test data, the 'uda-split' is used for unlabeled target training data,
    # the 'out-split' is used for validation data.
    in_splits = []  # in_splits = [source_training_data, target_test_data]
    out_splits = []  # out_splits = [source_val_data, target_val_data]
    uda_splits = []  # uda_splits = [target_training_data]
    for env_i, env in enumerate(dataset):
        uda = []
        if args.task == 'domain_adaptation':
            in_, out, uda = None, None, None
            if env_i in args.source_envs:
                out, in_ = misc.split_dataset(
                    env,
                    int(len(env) * args.holdout_fraction),
                    misc.seed_hash(args.trial_seed, env_i),
                )
                print(
                    'source train set size', len(in_), 'source val set size:', len(out)
                )
            elif env_i in args.target_envs:
                out, in_ = misc.split_dataset(
                    env,
                    int(len(env) * args.holdout_fraction),
                    misc.seed_hash(args.trial_seed, env_i),
                )
                uda, in_ = misc.split_dataset(
                    in_,
                    int(len(in_) * args.uda_holdout_fraction),
                    misc.seed_hash(args.trial_seed, env_i),
                )
                print('target train set size:', len(uda),
                    'target val set size:', len(out),
                    'target test set size:', len(in_),)
            if hparams['class_balanced'] and in_ is not None:
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

    if args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")

    # Domain adaptation for single source.
    if args.task == 'domain_adaptation':
        # Data augmentation for both source training data and target training data.
        for i, (env, env_weights) in enumerate(in_splits):
            if i in args.source_envs:
                env.underlying_dataset.transform = train_transform
            if i in args.target_envs:
                env.underlying_dataset.transform = test_transform
        for i, (env, env_weights) in enumerate(uda_splits):
            if i in args.target_envs:
                env.underlying_dataset.underlying_dataset.transform = train_transform
        for i, (env, env_weights) in enumerate(out_splits):
            if i in args.source_envs or i in args.target_envs:
                env.underlying_dataset.transform = test_transform
        train_loaders = [InfiniteDataLoader(dataset=env, weights=env_weights, 
                        batch_size=hparams['batch_size'], num_workers=dataset.N_WORKERS,)
                        for i, (env, env_weights) in enumerate(in_splits)
                        if i in args.source_envs]

        uda_loaders = [InfiniteDataLoader(dataset=env, weights=env_weights,
                        batch_size=hparams['batch_size'], num_workers=dataset.N_WORKERS,)
                        for i, (env, env_weights) in enumerate(uda_splits)
                        if i in args.target_envs]
    else:
        raise NotImplementedError

    eval_loaders = [
        FastDataLoader(dataset=env, batch_size=64, num_workers=dataset.N_WORKERS)
        for env, _ in (in_splits + out_splits + uda_splits)
        if env is not None]
    eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
    eval_loader_names = ['env{}_in'.format(i) for i in range(len(in_splits)) if in_splits[i][0] is not None]
    eval_loader_names += ['env{}_out'.format(i) for i in range(len(out_splits)) if out_splits[i][0] is not None]
    eval_loader_names += ['env{}_uda'.format(i) for i in range(len(uda_splits)) if uda_splits[i][0] is not None]

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    if args.task == 'domain_adaptation':
        num_domain = 2
    else:
        raise NotImplementedError
    algorithm = algorithm_class(
        dataset.input_shape,
        dataset.num_classes,
        num_domain,
        hparams,
    )
    algorithm.to(device)

    train_minibatches_iterator = zip(*train_loaders)
    uda_minibatches_iterator = zip(*uda_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ

    def save_checkpoint(filename):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.target_envs),
            "model_hparams": hparams,
            "model_dict": algorithm.state_dict()
        }
        torch.save(save_dict, os.path.join(args.output_dir, filename))

    last_results_keys = None
    pseudolabel_predictor = algorithm_class(
        dataset.input_shape,
        dataset.num_classes,
        num_domain,
        hparams,
    )
    pseudolabel_predictor.to(device)
    if (args.algorithm in ['AT', 'TRADES'] and hparams['pseudolabel']) or args.algorithm in ['SROUDA','DART']:
        # Initialize pseudolabel predictor by pretrained model weight.
        # Load pretrained model weight gained by training DANN on clean
        # source data with unlabeled target data.
        pseudolabel_predictor.load_state_dict(
            torch.load(
                os.path.join(
                    args.pretrain_model_dir,
                    str(args.dataset),
                    str(args.source_envs[0])+
                    str(args.target_envs[0]),
                    'model.pkl',
                )
            )['model_dict']
        )

    if args.pretrain == 1 or args.algorithm in ['SROUDA']:
        algorithm.load_state_dict(
            torch.load(
                os.path.join(
                    args.pretrain_model_dir,
                    str(args.dataset),
                    str(args.source_envs[0])+
                    str(args.target_envs[0]),
                    'model.pkl',
                )
            )['model_dict']
        )
    src_val = 'env{}_out'.format(args.source_envs[0])
    tgt_val = 'env{}_out'.format(args.target_envs[0])
    train_val_acc, test_val_acc, train_val_pgd_acc, test_val_pgd_acc = 0, 0, 0, 0

    for step in range(start_step, n_steps):
        step_start_time = time.time()
        if args.task == 'domain_adaptation':
            uda_device = [x.to(device) for x, _ in next(uda_minibatches_iterator)]
        else:
            uda_device = None
        minibatches_device = [
            (x.to(device), y.to(device))
            for x, y in next(train_minibatches_iterator)
        ]
        if args.algorithm in ['SROUDA']:
            step_vals, old_loss = algorithm.update(
                minibatches=minibatches_device,
                unlabeled=uda_device,
                eps=args.eps,
                atk_lr=args.atk_lr,
                atk_iter=args.atk_iter,
                pseudolabel_predictor=pseudolabel_predictor,
                metastep=0,
            )
            step_vals1 = pseudolabel_predictor.update(
                minibatches=minibatches_device,
                unlabeled=uda_device,
                eps=args.eps,
                atk_lr=args.atk_lr,
                atk_iter=args.atk_iter,
                pseudolabel_predictor=algorithm,
                metastep=1,
                old_loss=old_loss,
            )
        elif args.algorithm in ['ERM', 'DANN']:
            step_vals = algorithm.update(
                minibatches=minibatches_device,
                unlabeled=uda_device,
            )
        else:
            step_vals = algorithm.update(
                minibatches=minibatches_device,
                unlabeled=uda_device,
                source_type=args.source_type,
                eps=args.eps,
                atk_lr=args.atk_lr,
                atk_iter=args.atk_iter,
                pseudolabel_predictor=pseudolabel_predictor,
            )
        checkpoint_vals['step_time'].append(time.time() - step_start_time)

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        if (step % checkpoint_freq == 0) or (step == n_steps - 1):
            results = {
                'step': step,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            evals = zip(eval_loader_names, eval_loaders, eval_weights)
            for name, loader, _ in evals:
                if 'uda' in name:
                    continue
                acc = misc.accuracy(algorithm, loader, device)
                results[name + '_acc'] = acc
                # Evaluation on adversarial examples.
                if args.attack == 'pgd':
                    pgd_acc = misc.pgd_accuracy(
                        algorithm,
                        loader,
                        args.eps,
                        args.atk_lr,
                        args.atk_iter,
                        device,
                        1,
                    )
                    results[name + '_pgd_acc'] = pgd_acc
                    # Save the best model weight based on source val pgd accuracy.
                    if name == src_val:
                        if train_val_pgd_acc <= pgd_acc:
                            train_val_pgd_acc = pgd_acc
                            save_checkpoint('model_train_pgd_best.pkl')
                    # Save the best model weight based on target val pgd accuracy.
                    if name == tgt_val:
                        if test_val_pgd_acc <= pgd_acc:
                            test_val_pgd_acc = pgd_acc
                            save_checkpoint('model_test_pgd_best.pkl')
                # Save the best model weight based on source val clean accuracy.
                if name == src_val:
                    if train_val_acc <= acc:
                        train_val_acc = acc
                        save_checkpoint('model_train_best.pkl')
                # Save the best model weight based on target val pgd accuracy.
                if name == tgt_val:
                    if test_val_acc <= acc:
                        test_val_acc = acc
                        save_checkpoint('model_test_best.pkl')
                    if args.algorithm in ['DART']:
                        pseudolabel_predictor.load_state_dict(
                            torch.load(
                                os.path.join(
                                    args.output_dir,
                                    'model_test_best.pkl',
                                )
                            )['model_dict']
                        )

            results['mem_gb'] = torch.cuda.max_memory_allocated() / (
                1024.0 * 1024.0 * 1024.0
            )

            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=16)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys], colwidth=16)

            results.update({
                'hparams': hparams,
                'args': vars(args)
            })

            epochs_path = os.path.join(args.output_dir, 'results.jsonl')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")

            algorithm_dict = algorithm.state_dict()
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

            if args.save_model_every_checkpoint:
                save_checkpoint(f'model_step{step}.pkl')

    save_checkpoint('model.pkl')

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')
