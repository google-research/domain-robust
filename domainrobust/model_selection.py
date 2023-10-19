# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright 2023 Google LLC

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import itertools
import numpy as np

def get_test_records(records):
    """Given records with a common test env, get the test records (i.e. the
    records with *only* that single test env and no other test envs)"""
    return records.filter(lambda r: len(r['args']['target_envs']) == 1)

class SelectionMethod:
    """Abstract class whose subclasses implement strategies for model
    selection across hparams and timesteps."""

    def __init__(self):
        raise TypeError

    @classmethod
    def run_acc(self, run_records, task='domain_adaptation', attack='None'):
        """
        Given records from a run, return a {val_acc, test_acc} dict representing
        the best val-acc and corresponding test-acc for that run.
        """
        raise NotImplementedError

    @classmethod
    def hparams_accs(self, records, task='domain_adaptation', attack='None'):
        """
        Given all records from a single (dataset, algorithm, test env) pair,
        return a sorted list of (run_acc, records) tuples.
        """
        if attack == 'pgd':
            val_name = 'val_pgd_acc'
        else:
            val_name = 'val_acc'

        return (records.group('args.hparams_seed')
            .map(lambda _, run_records:
                (
                    self.run_acc(run_records, task, attack),
                    run_records
                )
            ).filter(lambda x: x[0] is not None)
            .sorted(key=lambda x: x[0][val_name])[::-1]
        )

    @classmethod
    def sweep_acc(self, records, task='domain_adaptation', attack='None'):
        """
        Given all records from a single (dataset, algorithm, test env) pair,
        return the mean test acc of the k runs with the top val accs.
        """
        _hparams_accs = self.hparams_accs(records, task, attack)
        if len(_hparams_accs):
            if attack == 'pgd':
                return [_hparams_accs[0][0]['test_acc'],_hparams_accs[0][0]['test_pgd_acc'], _hparams_accs[0][0]['path']]
            else:
                return [_hparams_accs[0][0]['test_acc'], _hparams_accs[0][0]['path']]
        else:
            return None

class OracleSelectionMethod(SelectionMethod):
    """Like Selection method which picks argmax(test_out_acc) across all hparams
    and checkpoints, but instead of taking the argmax over all
    checkpoints, we pick the last checkpoint, i.e. no early stopping."""
    name = "test-domain validation set (oracle)"

    @classmethod
    def _step_acc(self, record, task='domain_generalization', attack='None'):
        """Given a single record, return a {val_acc, test_acc} dict."""
        target_env = record['args']['target_envs'][0]
        val_env_keys = 'env{}_out_acc'.format(target_env)
        test_in_acc_key = 'env{}_in_acc'.format(target_env)
        if attack=='None':
            return {
                'val_acc': record[val_env_keys],
                'test_acc': record[test_in_acc_key],
                'lambda': record['hparams']['lambda'],
                'step': record['step'],
                'path': record['args']['output_dir'],
            }
        else:
            val_env_pgd_keys = 'env{}_out_pgd_acc'.format(target_env)
            test_in_pgd_acc_key = 'env{}_in_pgd_acc'.format(target_env)
            return {
                'val_acc': record[val_env_keys],
                'test_acc': record[test_in_acc_key],
                'val_pgd_acc': record[val_env_pgd_keys],
                'test_pgd_acc': record[test_in_pgd_acc_key],
                'lambda': record['hparams']['lambda'],
                'lambda1': record['hparams']['lambda1'],
                'step': record['step'],
                'path': record['args']['output_dir'],
            }

    @classmethod
    def run_acc(self, run_records, task='domain_adaptation', attack='None'):
        test_records = get_test_records(run_records)
        if not len(test_records):
            return None
        if attack == 'pgd':
            return test_records.map(lambda records: self._step_acc(records, task, attack)).argmax('val_pgd_acc')
        else:
            return test_records.map(lambda records: self._step_acc(records, task, attack)).argmax('val_acc')


class IIDAccuracySelectionMethod(SelectionMethod):
    """Picks argmax(mean(env_out_acc for env in source_envs))"""
    name = "training-domain validation set"

    @classmethod
    def _step_acc(self, record):
        """Given a single record, return a {val_acc, test_acc} dict."""
        test_env = record['args']['target_envs'][0]
        val_env_keys = []
        for i in itertools.count():
            if f'env{i}_out_acc' not in record:
                break
            if i != test_env:
                val_env_keys.append(f'env{i}_out_acc')
        test_in_acc_key = 'env{}_in_acc'.format(test_env)
        return {
            'val_acc': np.mean([record[key] for key in val_env_keys]),
            'test_acc': record[test_in_acc_key]
        }

    @classmethod
    def run_acc(self, run_records):
        test_records = get_test_records(run_records)
        if not len(test_records):
            return None
        return test_records.map(self._step_acc).argmax('val_acc')
