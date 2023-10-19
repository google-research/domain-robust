# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright 2023 Google LLC

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.


import numpy as np
from domainrobust.lib import misc


def _define_hparam(hparams, hparam_name, default_val, random_val_fn):
    hparams[hparam_name] = (hparams, hparam_name, default_val, random_val_fn)


def _hparams(algorithm, dataset, random_seed):
    """
    Global registry of hyperparams. Each entry is a (default, random) tuple.
    New algorithms / networks / etc. should add entries here.
    """
    SMALL_IMAGES = ['DIGIT']

    hparams = {}

    def _hparam(name, default_val, random_val_fn):
        """Define a hyperparameter. random_val_fn takes a RandomState and
        returns a random hyperparameter value."""
        assert(name not in hparams)
        random_state = np.random.RandomState(
            misc.seed_hash(random_seed, name)
        )
        hparams[name] = (default_val, random_val_fn(random_state))

    _hparam('resnet18', False, lambda r: False)
    _hparam('resnet_dropout', 0., lambda r: r.choice([0., 0.1, 0.5]))
    # TODO: class balanced disabled
    _hparam('class_balanced', False, lambda r: False)
    # TODO: nonlinear classifiers disabled
    _hparam('nonlinear_classifier', False,
            lambda r: bool(r.choice([False, False])))

    _hparam('lambda', 1.0, lambda r: 10**r.uniform(-1, 1))
    _hparam('lambda1', 1.0, lambda r: 10**r.uniform(-1, 1))
    _hparam('d_steps_per_g_step', 1, lambda r: int(2**r.uniform(0, 3)))
    _hparam('beta1', 0.9, lambda r: r.uniform(0, 0.9))
    _hparam('mlp_width', 1024, lambda r: 1024)
    _hparam('mlp_depth', 5, lambda r: 5)
    _hparam('mlp_dropout', 0., lambda r: 0.)

    if algorithm in ['AT', 'TRADES']:
        # Whether to use the pretrain model to predict the pseudolabel for the
        # target training samples or not. If True: target data with pseudolabel,
        # If False, source data only.
        _hparam('pseudolabel', False, lambda r: bool(r.choice([False, True])))

    # Dataset-specific hparam definitions.
    if dataset in SMALL_IMAGES:
        _hparam('lr_g', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5))
        _hparam('lr_d', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5))
        _hparam('weight_decay', 0., lambda r: 0.)
        _hparam('weight_decay_d', 0., lambda r: 10**r.uniform(-6, -3))
        _hparam('weight_decay_g', 0., lambda r: 0.)
        _hparam('batch_size', 128, lambda r: 128)
        _hparam('data_augmentation', False, lambda r: False)
    else:
        _hparam('lr_g', 5e-5, lambda r: 10**r.uniform(-5, -3.5))
        _hparam('lr_d', 5e-5, lambda r: 10**r.uniform(-5, -3.5))
        _hparam('weight_decay', 1e-4, lambda r: 10**r.uniform(-5, -2))
        _hparam('weight_decay_d', 1e-4, lambda r: 10**r.uniform(-5, -2))
        _hparam('weight_decay_g', 1e-4, lambda r: 10**r.uniform(-5, -2))
        _hparam('batch_size', 16, lambda r: 16)
        _hparam('data_augmentation', True, lambda r: True)

    return hparams

def default_hparams(algorithm, dataset):
    return {a: b for a, (b, c) in _hparams(algorithm, dataset, 0).items()}


def random_hparams(algorithm, dataset, seed):
    return {a: c for a, (b, c) in _hparams(algorithm, dataset, seed).items()}

