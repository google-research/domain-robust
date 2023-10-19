# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright 2023 Google LLC

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from domainrobust import networks
from domainrobust.third_party.cleverhans.pgd import projected_gradient_descent

ALGORITHMS = [
    'ERM',
    'DANN',
    'AT',
    'TRADES',
    'ATUDA',
    'ARTUDA',
    'SROUDA',
    'DART',
]


def get_algorithm_class(algorithm_name):
  """Return the algorithm class with the given name."""
  if algorithm_name not in globals():
    raise NotImplementedError('Algorithm not found: {}'.format(algorithm_name))
  return globals()[algorithm_name]

class Algorithm(torch.nn.Module):
  """A subclass of Algorithm implements a domain adaptation algorithm.
    Subclasses should implement the following: 
    - update()
    - predict()
  """
  def __init__(self, input_shape, num_classes, num_domains, hparams):
    super(Algorithm, self).__init__()
    self.hparams = hparams

  def update(self, minibatches, unlabeled=None):
    """
    Perform one update step.
    minibatches: given a list of (x, y) tuples for source domain.
    unlabeled: a list of unlabeled minibatches from the target domain,
              when task is domain_adaptation.
    """
    raise NotImplementedError

  def predict(self, x):
    raise NotImplementedError

class AbstractDANN(Algorithm):
  """Domain-Adversarial Neural Networks (abstract class)"""

  def __init__(
      self,
      input_shape,
      num_classes,
      num_domains,
      hparams,
      conditional,
      class_balance,
  ):
    super(AbstractDANN, self).__init__(
        input_shape, num_classes, num_domains, hparams
    )

    self.register_buffer('update_count', torch.tensor([0]))
    self.conditional = conditional
    self.class_balance = class_balance

    # Architecture
    self.featurizer = networks.Featurizer(input_shape, self.hparams)
    self.classifier = networks.Classifier(
        self.featurizer.n_outputs,
        num_classes,
        self.hparams['nonlinear_classifier'],
    )
    self.discriminator = networks.MLP(
        self.featurizer.n_outputs, num_domains, self.hparams
    )
    self.class_embeddings = nn.Embedding(num_classes, self.featurizer.n_outputs)

    # Optimizers
    self.disc_opt = torch.optim.Adam(
        (
            list(self.discriminator.parameters())
            + list(self.class_embeddings.parameters())
        ),
        lr=self.hparams['lr_d'],
        weight_decay=self.hparams['weight_decay_d'],
        betas=(self.hparams['beta1'], 0.9),
    )

    self.gen_opt = torch.optim.Adam(
        (
            list(self.featurizer.parameters())
            + list(self.classifier.parameters())
        ),
        lr=self.hparams['lr_g'],
        weight_decay=self.hparams['weight_decay_g'],
        betas=(self.hparams['beta1'], 0.9),
    )

  def update(self, minibatches, unlabeled=None):
    device = 'cuda' if minibatches[0][0].is_cuda else 'cpu'
    self.update_count += 1
    all_x = torch.cat([x for x, y in minibatches])
    all_y = torch.cat([y for x, y in minibatches])
    all_z = self.featurizer(all_x)
    if self.conditional:
      disc_input_source = all_z + self.class_embeddings(all_y)
    else:
      disc_input_source = all_z
    if unlabeled:
      # Domain adaptation
      all_target = torch.cat(list(unlabeled))
      disc_input_target = self.featurizer(all_target)
      disc_input_combined = torch.cat((disc_input_source, disc_input_target), dim=0)

      disc_out = self.discriminator(disc_input_combined)
      disc_labels = torch.cat([
          torch.full(
              (all_x.shape[0],),
              1,
              dtype=torch.int64,
              device=device,
          ),
          torch.full(
              (all_target.shape[0],),
              0,
              dtype=torch.int64,
              device=device,
          ),
      ])
      disc_loss = F.cross_entropy(disc_out, disc_labels)
    else:
      raise NotImplementedError

    d_steps_per_g = self.hparams['d_steps_per_g_step']
    if self.update_count.item() % (1 + d_steps_per_g) < d_steps_per_g:
      self.disc_opt.zero_grad()
      disc_loss.backward()
      self.disc_opt.step()
      return {'disc_loss': disc_loss.item()}
    else:
      all_preds = self.classifier(all_z)
      classifier_loss = F.cross_entropy(all_preds, all_y)
      gen_loss = classifier_loss + (self.hparams['lambda'] * -disc_loss)
      self.disc_opt.zero_grad()
      self.gen_opt.zero_grad()
      gen_loss.backward()
      self.gen_opt.step()
      return {'gen_loss': gen_loss.item()}

  def predict(self, x):
    return self.classifier(self.featurizer(x))


class ERM(AbstractDANN):
  """Empirical Risk Minimization (ERM)"""

  def __init__(self, input_shape, num_classes, num_domains, hparams):
    super(ERM, self).__init__(
        input_shape,
        num_classes,
        num_domains,
        hparams,
        conditional=False,
        class_balance=False,
    )

  def update(self, minibatches, unlabeled=None):
    all_x = torch.cat([x for x, y in minibatches])
    all_y = torch.cat([y for x, y in minibatches])
    loss = F.cross_entropy(self.predict(all_x), all_y)
    self.gen_opt.zero_grad()
    loss.backward()
    self.gen_opt.step()
    return {'loss': loss.item()}


class DANN(AbstractDANN):
  """Unconditional DANN"""

  def __init__(self, input_shape, num_classes, num_domains, hparams):
    super(DANN, self).__init__(
        input_shape,
        num_classes,
        num_domains,
        hparams,
        conditional=False,
        class_balance=False,
    )


class DART(AbstractDANN):
  """
  Proposed algorithm.
  """
  def __init__(self, input_shape, num_classes, num_domains, hparams):
    super(DART, self).__init__(
        input_shape,
        num_classes,
        num_domains,
        hparams,
        conditional=False,
        class_balance=False,
    )

  def update(
      self,
      minibatches,
      unlabeled=None,
      source_type='clean',
      eps=0.0,
      atk_lr=0.0,
      atk_iter=0,
      pseudolabel_predictor=None
  ):
    """
    :param minibatches: labeled source data. List containing tensor tuples.
    :param unlabeled: unlabeled target data. Tensor list.
    :param source_type: the type of source for the algorithm.
              Option: clean, adv, kl.
    :param eps: the size of perturbation.
    :param atk_lr: step size for each attack iteration.
    :param atk_iter: Number of attack iterations.
    :param pseudolabel_predictor: the model that used to provide pseudo-label 
              for target domain.
    """
    device = 'cuda' if minibatches[0][0].is_cuda else 'cpu'
    self.update_count += 1
    all_x = torch.cat([x for x, y in minibatches])
    all_y = torch.cat([y for x, y in minibatches])
    all_target = torch.cat(list(unlabeled))
    if source_type == 'clean':
      adv_source = all_x
    elif source_type == 'adv':
      adv_source = projected_gradient_descent(
          model_featurizer=self.featurizer,
          model_classifier=self.classifier,
          model_discriminator=self.discriminator,
          hparams=self.hparams,
          x=all_x,
          eps=eps,
          eps_iter=atk_lr,
          nb_iter=atk_iter,
          norm=np.inf,
          clip_min=0,
          clip_max=1,
          y=all_y,
          targeted=False,
          rand_init=True,
          rand_minmax=None,
          adv_type='adv',
          src=None,
          sanity_checks=True,
      )
    elif source_type == 'kl':
      adv_source = projected_gradient_descent(
          model_featurizer=self.featurizer,
          model_classifier=self.classifier,
          model_discriminator=self.discriminator,
          hparams=self.hparams,
          x=all_x,
          eps=eps,
          eps_iter=atk_lr,
          nb_iter=atk_iter,
          norm=np.inf,
          clip_min=0,
          clip_max=1,
          y=None,
          targeted=False,
          rand_init=True,
          rand_minmax=None,
          adv_type='kl',
          src=None,
          sanity_checks=True
      )
    else:
      raise NotImplementedError

    adv_target = projected_gradient_descent(
        model_featurizer=self.featurizer,
        model_classifier=self.classifier,
        model_discriminator=self.discriminator,
        hparams=self.hparams,
        x=all_target,
        eps=eps,
        eps_iter=atk_lr,
        nb_iter=atk_iter,
        norm=np.inf,
        clip_min=0,
        clip_max=1,
        y=None,
        targeted=False,
        rand_init=True,
        rand_minmax=None,
        adv_type='dannmaxdivconst',
        src=adv_source,
        sanity_checks=True,
    )

    disc_input_source = self.featurizer(adv_source)
    disc_input_target = self.featurizer(adv_target)
    disc_input_combined = torch.cat(
        (disc_input_source, disc_input_target), dim=0
    )

    disc_out = self.discriminator(disc_input_combined)
    disc_labels = torch.cat([
        torch.full(
            (all_x.shape[0],),
            1,
            dtype=torch.int64,
            device=device,
        ),
        torch.full(
            (all_target.shape[0],),
            0,
            dtype=torch.int64,
            device=device,
        ),
    ])
    disc_loss = F.cross_entropy(disc_out, disc_labels)
    d_steps_per_g = self.hparams['d_steps_per_g_step']
    if (
        self.update_count.item() % (1 + d_steps_per_g) < d_steps_per_g
    ):
      self.disc_opt.zero_grad()
      disc_loss.backward()
      self.disc_opt.step()
      return {'disc_loss': disc_loss.item()}
    else:
      classifier_loss = F.cross_entropy(
          self.classifier(disc_input_source), all_y
      )
      _, pseudo_tgt_y = torch.max(pseudolabel_predictor.predict(all_target), 1)
      classifier_loss += self.hparams['lambda1'] * F.cross_entropy(
          self.classifier(disc_input_target), pseudo_tgt_y
      )
      gen_loss = classifier_loss - self.hparams['lambda'] * disc_loss
      self.disc_opt.zero_grad()
      self.gen_opt.zero_grad()
      gen_loss.backward()
      self.gen_opt.step()
      return {
          'gen_loss': gen_loss.item(),
      }


class ATUDA(AbstractDANN):
  """
  Defined baseline, first transfer the source to adversarial examples, 
  then apply standard UDA (DANN) on adversarial source and clean target.
  """
  def __init__(self, input_shape, num_classes, num_domains, hparams):
    super(ATUDA, self).__init__(
        input_shape,
        num_classes,
        num_domains,
        hparams,
        conditional=False,
        class_balance=False,
    )

  def update(
      self,
      minibatches,
      unlabeled=None,
      source_type='adv',
      eps=0.0,
      atk_lr=0.0,
      atk_iter=0,
      pseudolabel_predictor=None
  ):
    """
    :param minibatches: labeled source data. List containing tensor tuples.
    :param unlabeled: unlabeled target data. Tensor list.
    :param eps: the size of perturbation.
    :param atk_lr: step size for each attack iteration.
    :param atk_iter: Number of attack iterations.
    :param pseudolabel_predictor: the model that used to provide pseudo-label 
              for target domain.
    """
    device = 'cuda' if minibatches[0][0].is_cuda else 'cpu'
    self.update_count += 1
    all_x = torch.cat([x for x, y in minibatches])
    all_y = torch.cat([y for x, y in minibatches])
    all_target = torch.cat(list(unlabeled))
    adv_source = projected_gradient_descent(
        model_featurizer=self.featurizer,
        model_classifier=self.classifier,
        model_discriminator=self.discriminator,
        hparams=self.hparams,
        x=all_x,
        eps=eps,
        eps_iter=atk_lr,
        nb_iter=atk_iter,
        norm=np.inf,
        clip_min=0,
        clip_max=1,
        y=all_y,
        targeted=False,
        rand_init=True,
        rand_minmax=None,
        adv_type='adv',
        src=None,
        sanity_checks=True,
    )

    disc_input_source = self.featurizer(adv_source)
    disc_input_target = self.featurizer(all_target)
    disc_input_combined = torch.cat(
        (disc_input_source, disc_input_target), dim=0
    )

    disc_out = self.discriminator(disc_input_combined)
    disc_labels = torch.cat([
        torch.full(
            (all_x.shape[0],),
            1,
            dtype=torch.int64,
            device=device,
        ),
        torch.full(
            (all_target.shape[0],),
            0,
            dtype=torch.int64,
            device=device,
        ),
    ])
    disc_loss = F.cross_entropy(disc_out, disc_labels)
    d_steps_per_g = self.hparams['d_steps_per_g_step']
    if self.update_count.item() % (1 + d_steps_per_g) < d_steps_per_g:
      self.disc_opt.zero_grad()
      disc_loss.backward()
      self.disc_opt.step()
      return {'disc_loss': disc_loss.item()}
    else:
      classifier_loss = F.cross_entropy(
          self.classifier(self.featurizer(adv_source)), all_y
      )
      gen_loss = classifier_loss - self.hparams['lambda'] * disc_loss
      self.disc_opt.zero_grad()
      self.gen_opt.zero_grad()
      gen_loss.backward()
      self.gen_opt.step()
      return {
          'gen_loss': gen_loss.item(),
      }


class AT(AbstractDANN):
  """
  https://arxiv.org/abs/1706.06083
  """
  def __init__(self, input_shape, num_classes, num_domains, hparams):
    super(AT, self).__init__(
        input_shape,
        num_classes,
        num_domains,
        hparams,
        conditional=False,
        class_balance=False,
    )

  def update(
      self,
      minibatches,
      unlabeled=None,
      source_type='clean',
      eps=0.0,
      atk_lr=0.0,
      atk_iter=0,
      pseudolabel_predictor=None
  ):
    self.update_count += 1
    if self.hparams['pseudolabel']:
      all_x = torch.cat([x for x in unlabeled])
      _, all_y = torch.max(pseudolabel_predictor.predict(all_x), 1)
    else:
      all_x = torch.cat([x for x, y in minibatches])
      all_y = torch.cat([y for x, y in minibatches])
    adv_source = projected_gradient_descent(
        model_featurizer=self.featurizer,
        model_classifier=self.classifier,
        model_discriminator=None,
        hparams=self.hparams,
        x=all_x,
        eps=eps,
        eps_iter=atk_lr,
        nb_iter=atk_iter,
        norm=np.inf,
        clip_min=0,
        clip_max=1,
        y=all_y,
        targeted=False,
        rand_init=True,
        rand_minmax=None,
        adv_type='adv',
        src=None,
        sanity_checks=True,
    )

    disc_input_source = self.featurizer(adv_source)
    all_preds = self.classifier(disc_input_source)
    gen_loss = F.cross_entropy(all_preds, all_y)
    self.gen_opt.zero_grad()
    gen_loss.backward()
    self.gen_opt.step()
    return {
        'gen_loss': gen_loss.item(),
    }

class TRADES(AbstractDANN):
  """
  https://arxiv.org/abs/1901.08573
  """
  def __init__(self, input_shape, num_classes, num_domains, hparams):
    super(TRADES, self).__init__(
        input_shape,
        num_classes,
        num_domains,
        hparams,
        conditional=False,
        class_balance=False,
    )

  def update(
      self,
      minibatches,
      unlabeled=None,
      source_type='clean',
      eps=0.0,
      atk_lr=0.0,
      atk_iter=0,
      pseudolabel_predictor=None
  ):
    self.update_count += 1
    if self.hparams['pseudolabel']:
      all_x = torch.cat([x for x in unlabeled])
      _, all_y = torch.max(pseudolabel_predictor.predict(all_x), 1)
    else:
      all_x = torch.cat([x for x, y in minibatches])
      all_y = torch.cat([y for x, y in minibatches])
    kl = nn.KLDivLoss(reduction='none')
    classifier_loss = F.cross_entropy(
        self.classifier(self.featurizer(all_x)), all_y
    )
    adv_x = projected_gradient_descent(
        model_featurizer=self.featurizer,
        model_classifier=self.classifier,
        model_discriminator=self.discriminator,
        hparams=self.hparams,
        x=all_x,
        eps=eps,
        eps_iter=atk_lr,
        nb_iter=atk_iter,
        norm=np.inf,
        clip_min=0,
        clip_max=1,
        y=all_y,
        targeted=False,
        rand_init=True,
        rand_minmax=None,
        adv_type='kl',
        src=None,
        sanity_checks=True,
    )
    gen_loss = classifier_loss + (
        self.hparams['lambda1']
        * (1.0 / self.hparams['batch_size'])
        * torch.sum(
            torch.sum(
                kl(
                    F.log_softmax(
                        self.classifier(self.featurizer(adv_x)), dim=1
                    ),
                    F.softmax(self.classifier(self.featurizer(all_x)), dim=1),
                )
            )
        )
    )
    self.gen_opt.zero_grad()
    gen_loss.backward()
    self.gen_opt.step()
    return {
        'gen_loss': gen_loss.item(),
    }


class ARTUDA(AbstractDANN):
  """https://arxiv.org/abs/2202.09300"""
  def __init__(self, input_shape, num_classes, num_domains, hparams):
    super(ARTUDA, self).__init__(
        input_shape,
        num_classes,
        num_domains,
        hparams,
        conditional=False,
        class_balance=False,
    )

  def update(
      self,
      minibatches,
      unlabeled=None,
      source_type='clean',
      eps=0.0,
      atk_lr=0.0,
      atk_iter=0,
      pseudolabel_predictor=None
  ):
    device = 'cuda' if minibatches[0][0].is_cuda else 'cpu'
    self.update_count += 1
    all_x = torch.cat([x for x, y in minibatches])
    all_y = torch.cat([y for x, y in minibatches])
    all_target = torch.cat(list(unlabeled))
    adv_target = projected_gradient_descent(
        model_featurizer=self.featurizer,
        model_classifier=self.classifier,
        model_discriminator=self.discriminator,
        hparams=self.hparams,
        x=all_target,
        eps=eps,
        eps_iter=atk_lr,
        nb_iter=atk_iter,
        norm=np.inf,
        clip_min=0,
        clip_max=1,
        y=None,
        targeted=False,
        rand_init=True,
        rand_minmax=None,
        adv_type='kl',
        src=None,
        sanity_checks=True,
    )
    disc_input_source = self.featurizer(all_x)
    disc_input_target = self.featurizer(adv_target)
    disc_input_combined = torch.cat(
        (disc_input_source, disc_input_target), dim=0
    )

    disc_out = self.discriminator(disc_input_combined)
    disc_labels = torch.cat([
        torch.full(
            (all_x.shape[0],),
            1,
            dtype=torch.int64,
            device=device,
        ),
        torch.full(
            (all_target.shape[0],),
            0,
            dtype=torch.int64,
            device=device,
        ),
    ])
    disc_loss = F.cross_entropy(disc_out, disc_labels)
    disc_loss += F.cross_entropy(
        self.discriminator(
            torch.cat(
                (self.featurizer(all_x), self.featurizer(all_target)), dim=0
            )
        ),
        disc_labels,
    )
    d_steps_per_g = self.hparams['d_steps_per_g_step']
    if (
        self.update_count.item() % (1 + d_steps_per_g) < d_steps_per_g
    ):
      self.disc_opt.zero_grad()
      disc_loss.backward()
      self.disc_opt.step()
      return {'disc_loss': disc_loss.item()}
    else:
      kl = nn.KLDivLoss(reduction='none')
      classifier_loss = F.cross_entropy(
          self.classifier(self.featurizer(all_x)), all_y
      )
      classifier_loss += (
          self.hparams['lambda1']
          * (1.0 / self.hparams['batch_size'])
          * torch.sum(
              torch.sum(
                  kl(
                      F.log_softmax(
                          self.classifier(self.featurizer(adv_target)), dim=1
                      ),
                      F.softmax(
                          self.classifier(self.featurizer(all_target)), dim=1
                      ),
                  )
              )
          )
      )
      gen_loss = classifier_loss - self.hparams['lambda'] * disc_loss
      self.disc_opt.zero_grad()
      self.gen_opt.zero_grad()
      gen_loss.backward()
      self.gen_opt.step()
      return {
          'gen_loss': gen_loss.item(),
      }


class SROUDA(AbstractDANN):
  """
  https://arxiv.org/abs/2212.05917
  """
  def __init__(self, input_shape, num_classes, num_domains, hparams):
    super(SROUDA, self).__init__(
        input_shape,
        num_classes,
        num_domains,
        hparams,
        conditional=False,
        class_balance=False,
    )

  def update(
      self,
      minibatches,
      unlabeled=None,
      eps=0.0,
      atk_lr=0.0,
      atk_iter=0,
      pseudolabel_predictor=None,
      metastep=0,
      old_loss=None,
  ):
    self.update_count += 1
    all_x = torch.cat([x for x, y in minibatches])
    all_y = torch.cat([y for x, y in minibatches])
    all_target = torch.cat(list(unlabeled))
    if metastep == 0:
      _, pseudo_target_y = torch.max(
          pseudolabel_predictor.predict(all_target), 1
      )
      adv_target = projected_gradient_descent(
          model_featurizer=self.featurizer,
          model_classifier=self.classifier,
          model_discriminator=self.discriminator,
          hparams=self.hparams,
          x=all_target,
          eps=eps,
          eps_iter=atk_lr,
          nb_iter=atk_iter,
          norm=np.inf,
          clip_min=0,
          clip_max=1,
          y=pseudo_target_y,
          targeted=False,
          rand_init=True,
          rand_minmax=None,
          adv_type='adv',
          src=None,
          sanity_checks=True,
      )
      gen_loss = F.cross_entropy(
          self.classifier(self.featurizer(adv_target)), pseudo_target_y
      )
      old_loss = F.cross_entropy(
          self.classifier(self.featurizer(all_x)).detach(), all_y
      )
    elif metastep == 1:
      with torch.no_grad():
        soft_pseudo_label = torch.softmax(
            self.classifier(self.featurizer(all_target)).detach(), dim=-1
        )
      max_probs, _ = torch.max(soft_pseudo_label, dim=-1)
      mask = max_probs.ge(0.95).float()
      t_loss_u = torch.mean(
          -(
              soft_pseudo_label
              * torch.log_softmax(
                  self.classifier(self.featurizer(all_target)), dim=-1
              )
          ).sum(dim=-1)
          * mask
      )
      with torch.no_grad():
        new_loss = F.cross_entropy(
            pseudolabel_predictor.predict(all_x).detach(),
            all_y,
        )
      _, pseudo_tgt_y = torch.max(
          self.classifier(self.featurizer(all_target)), 1
      )
      gen_loss = (
          F.cross_entropy(self.classifier(self.featurizer(all_x)), all_y)
          + t_loss_u
          + (new_loss - old_loss)
          * F.cross_entropy(
              self.classifier(self.featurizer(all_target)), pseudo_tgt_y
          )
      )
    else:
      raise NotImplementedError
    self.gen_opt.zero_grad()
    gen_loss.backward()
    self.gen_opt.step()
    if metastep == 0:
      return {'gen_loss': gen_loss.item(),}, old_loss
    else:
      return {'gen_loss': gen_loss.item(),}