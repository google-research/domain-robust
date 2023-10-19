# Copyright (c) 2019 Google Inc., OpenAI and Pennsylvania State University
# Copyright 2023 Google LLC

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.


import numpy as np
import torch
import torch.nn.functional as F

def clip_eta(eta, norm, eps):
  """PyTorch implementation of the clip_eta in utils_tf.

  :param eta: Tensor
  :param norm: np.inf, 1, or 2
  :param eps: float
  """
  if norm not in [np.inf, 1, 2]:
    raise ValueError("norm must be np.inf, 1, or 2.")

  avoid_zero_div = torch.tensor(1e-12, dtype=eta.dtype, device=eta.device)
  reduc_ind = list(range(1, len(eta.size())))
  if norm == np.inf:
    eta = torch.clamp(eta, -eps, eps)
  else:
    if norm == 1:
      raise NotImplementedError("L1 clip is not implemented.")
    elif norm == 2:
      norm = torch.sqrt(
          torch.max(
              avoid_zero_div, torch.sum(eta**2, dim=reduc_ind, keepdim=True)
          )
      )
    factor = torch.min(
        torch.tensor(1.0, dtype=eta.dtype, device=eta.device), eps / norm
    )
    eta *= factor
  return eta


def optimize_linear(grad, eps, norm=np.inf):
  """Solves for the optimal input to a linear function under a norm constraint.

  Optimal_perturbation = argmax_{eta, ||eta||_{norm} < eps} dot(eta, grad)

  :param grad: Tensor, shape (N, d_1, ...). Batch of gradients
  :param eps: float. Scalar specifying size of constraint region
  :param norm: np.inf, 1, or 2. Order of norm constraint.
  :returns: Tensor, shape (N, d_1, ...). Optimal perturbation
  """

  red_ind = list(range(1, len(grad.size())))
  avoid_zero_div = torch.tensor(1e-12, dtype=grad.dtype, device=grad.device)
  if norm == np.inf:
    # Take sign of gradient
    optimal_perturbation = torch.sign(grad)
  elif norm == 1:
    sign = torch.sign(grad)
    red_ind = list(range(1, len(grad.size())))
    abs_grad = torch.abs(grad)
    ori_shape = [1] * len(grad.size())
    ori_shape[0] = grad.size(0)

    max_abs_grad, _ = torch.max(abs_grad.view(grad.size(0), -1), 1)
    max_mask = abs_grad.eq(max_abs_grad.view(ori_shape)).to(torch.float)
    num_ties = max_mask
    for red_scalar in red_ind:
      num_ties = torch.sum(num_ties, red_scalar, keepdim=True)
    optimal_perturbation = sign * max_mask / num_ties
    # TODO integrate below to a test file
    # check that the optimal perturbations have been correctly computed
    opt_pert_norm = optimal_perturbation.abs().sum(dim=red_ind)
    assert torch.all(opt_pert_norm == torch.ones_like(opt_pert_norm))
  elif norm == 2:
    square = torch.max(
        avoid_zero_div, torch.sum(grad**2, red_ind, keepdim=True)
    )
    optimal_perturbation = grad / torch.sqrt(square)
    # TODO integrate below to a test file
    # check that the optimal perturbations have been correctly computed
    opt_pert_norm = (
        optimal_perturbation.pow(2).sum(dim=red_ind, keepdim=True).sqrt()
    )
    one_mask = (square <= avoid_zero_div).to(torch.float) * opt_pert_norm + (
        square > avoid_zero_div
    ).to(torch.float)
    assert torch.allclose(opt_pert_norm, one_mask, rtol=1e-05, atol=1e-08)
  else:
    raise NotImplementedError(
        "Only L-inf, L1 and L2 norms are currently implemented."
    )

  # Scale perturbation to be the solution for the norm=eps rather than
  # norm=1 problem
  scaled_perturbation = eps * optimal_perturbation
  return scaled_perturbation


def fast_gradient_method(
    model_featurizer,
    model_classifier,
    model_discriminator,
    hparams,
    x,
    eps,
    norm,
    clip_min=None,
    clip_max=None,
    y=None,
    targeted=False,
    adv_type='adv',
    src=None,
    x_natural=None,
    sanity_checks=False,
):
  """PyTorch implementation of the Fast Gradient Method.

  :param model_featurizer: a callable that takes an input tensor and returns the feature tensor.
  :param model_classifier, model_discriminator: a callable that takes a feature tensor and returns the model logits.
  :param hparams: the hyperparameters of the algorithms.
  :param x: input tensor for the target domain.
  :param eps: epsilon (input variation parameter); see https://arxiv.org/abs/1412.6572. 
  :param norm: Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2. 
  :param clip_min: (optional) float. Minimum float value for adversarial example components.
  :param clip_max: (optional) float. Maximum float value for adversarial example components. 
  :param y: (optional) Tensor with true labels. If targeted is true, then provide the target label. 
            Otherwise, only provide this parameter if you'd like to use true labels when crafting 
            adversarial samples. Otherwise, model predictions are used as labels to avoid the 
            "label leaking" effect (explained in this paper: https://arxiv.org/abs/1611.01236). Default is None.
  :param targeted: (optional) bool. Is the attack targeted or untargeted?
            Untargeted, the default, will try to make the label incorrect.
            Targeted will instead try to move in the direction of being more
            like y.
  :param adv_type: type of adversarial examples. Possible options: adv, danntarget, trades, dannmaxdivconst
            adv means generate adversarial examples by maximizing the CE loss(x,y).
            danntarget means generate adversarial examples by minimizing the CE loss([x_src,x],[0,1])
            trades means generate adversarial examples by maximizing the KL divergence loss.
            dannmaxdivconst means generate adversarial examples by maximizing 
                lambda -CE loss([x_src,x],[0,1])+lambda_1 CE loss(f(x),f(x_natural))
            Default is adv.
  :param src: the source examples. Default is None.
  :param x_natural: the clean examples x without adding perturbation. This is used when adv_type is trades.
            Default is None.
  :param sanity_checks: bool, if True, include asserts (Turn them off to use 
            less runtime memory or for unit tests that intentionally pass strange input)
  :return: a tensor of the adversarial examples.
  """
  if norm not in [np.inf, 1, 2]:
    raise ValueError(
        "Norm order must be either np.inf, 1, or 2, got {} instead.".format(
            norm
        )
    )
  if eps < 0:
    raise ValueError(
        "eps must be greater than or equal to 0, got {} instead".format(eps)
    )
  if eps == 0:
    return x
  if clip_min is not None and clip_max is not None:
    if clip_min > clip_max:
      raise ValueError(
          "clip_min must be less than or equal to clip_max, got clip_min={} and"
          " clip_max={}".format(clip_min, clip_max)
      )

  asserts = []

  # If a data range was specified, check that the input was in that range.
  if clip_min is not None:
    assert_ge = torch.all(
        torch.ge(x, torch.tensor(clip_min, device=x.device, dtype=x.dtype))
    )
    asserts.append(assert_ge.detach().cpu())

  if clip_max is not None:
    assert_le = torch.all(
        torch.le(x, torch.tensor(clip_max, device=x.device, dtype=x.dtype))
    )
    asserts.append(assert_le.detach().cpu())

  # x needs to be a leaf variable, of floating point type and have requires_grad
  # being True for its grad to be computed and stored properly in a backward call.
  x = x.clone().detach().to(torch.float).requires_grad_(True)
  x_natural = x_natural.clone().detach().to(torch.float).requires_grad_(False)
  if y is None:
    # Using model predictions as ground truth to avoid label leaking.
    _, y = torch.max(model_classifier(model_featurizer(x)), 1)

  # Compute loss.
  if adv_type == 'adv':
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(model_classifier(model_featurizer(x)), y)
    # If attack is targeted, minimize loss of target label rather than maximize
    # loss of correct label.
    if targeted:
      loss = -loss
  elif adv_type == "trades":
    loss_fn = torch.nn.KLDivLoss(size_average=False)
    loss = loss_fn(
        F.log_softmax(model_classifier(model_featurizer(x)), dim=1),
        F.softmax(model_classifier(model_featurizer(x_natural)), dim=1),
    )
  elif adv_type == "dannmaxdivconst":
    if src is None:
      raise NotImplementedError
    src = src.detach()
    loss_fn = torch.nn.CrossEntropyLoss()
    disc_out = model_discriminator(
        torch.cat((model_featurizer(src), model_featurizer(x)), dim=0)
    )
    disc_labels = torch.cat([
        torch.full(
            (src.shape[0],),
            1,
            dtype=torch.int64,
            device=src.device,
        ),
        torch.full(
            (x.shape[0],),
            0,
            dtype=torch.int64,
            device=x.device,
        ),
    ])
    _, pseudo_tgt_y = torch.max(model_classifier(model_featurizer(x_natural)), 1)
    loss = -hparams["lambda"] * loss_fn(disc_out, disc_labels) + hparams["lambda1"] *loss_fn(
        model_classifier(model_featurizer(x)), pseudo_tgt_y
    )
  elif adv_type == "danntarget":
    if src is None:
      raise NotImplementedError
    src = src.detach()
    loss_fn = torch.nn.CrossEntropyLoss()
    disc_out = model_discriminator(
        torch.cat((model_featurizer(src), model_featurizer(x)), dim=0)
    )
    disc_labels = torch.cat([
        torch.full(
            (src.shape[0],),
            1,
            dtype=torch.int64,
            device=src.device,
        ),
        torch.full(
            (x.shape[0],),
            0,
            dtype=torch.int64,
            device=x.device,
        ),
    ])
    loss = -loss_fn(disc_out, disc_labels)
  else:
    raise NotImplementedError

  # Define gradient of loss wrt input.
  loss.backward()
  optimal_perturbation = optimize_linear(x.grad, eps, norm)

  # Add perturbations to the original examples to obtain adversarial examples.
  adv_x = x + optimal_perturbation
  # If clipping is needed, reset all values outside of [clip_min, clip_max].
  if (clip_min is not None) or (clip_max is not None):
    if clip_min is None or clip_max is None:
      raise ValueError(
          "One of clip_min and clip_max is None but we don't currently support"
          " one-sided clipping"
      )
    adv_x = torch.clamp(adv_x, clip_min, clip_max)

  if sanity_checks:
    assert np.all(asserts)
  return adv_x


def projected_gradient_descent(
    model_featurizer,
    model_classifier,
    model_discriminator,
    hparams,
    x,
    eps,
    eps_iter,
    nb_iter,
    norm,
    clip_min=None,
    clip_max=None,
    y=None,
    targeted=False,
    rand_init=True,
    rand_minmax=None,
    adv_type='adv',
    src=None,
    sanity_checks=True,
):
  """This class implements either the Basic Iterative Method (Kurakin et al.

  2016) when rand_init is set to False or the Madry et al. (2017) method if
  rand_init is set to True. Paper link (Kurakin et al. 2016):
  https://arxiv.org/pdf/1607.02533.pdf Paper link (Madry et al. 2017):
  https://arxiv.org/pdf/1706.06083.pdf 
  :param model_featurizer: a callable that takes an input tensor and returns the feature tensor.
  :param model_classifier, model_discriminator: a callable that takes an feature tensor and returns the model logits.
  :param hparams: the hyperparameters of the algorithms.
  :param x: input tensor for the target domain. 
  :param eps: epsilon (input variation parameter); see https://arxiv.org/abs/1412.6572. 
  :param eps_iter: step size for each attack iteration.
  :param nb_iter: Number of attack iterations.
  :param norm: Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2. 
  :param clip_min: (optional) float. Minimum float value for adversarial example components.
  :param clip_max: (optional) float. Maximum float value for adversarial example components. 
  :param y: (optional) Tensor with true labels. If targeted is true, then provide the target label. 
            Otherwise, only provide this parameter if you'd like to use true labels when crafting 
            adversarial samples. Otherwise, model predictions are used as labels to avoid the 
            "label leaking" effect (explained in this paper: https://arxiv.org/abs/1611.01236). Default is None.
  :param targeted: (optional) bool. Is the attack targeted or untargeted?
            Untargeted, the default, will try to make the label incorrect.
            Targeted will instead try to move in the direction of being more
            like y.
  :param adv_type: type of adversarial examples. Possible options: adv, danntarget, trades.
            adv means generate adversarial examples by maximizing the CE loss(x,y).
            danntarget means generate adversarial examples by minimizing the CE loss([x_src,x],[0,1])
            trades means generate adversarial examples by maximizing the KL divergence loss.
            Default is adv.
  :param rand_init: (optional) bool. Whether to start the attack from a randomly perturbed x.
  :param rand_minmax: (optional) bool. Support of the continuous uniform distribution from
            which the random perturbation on x was drawn. Effective only when rand_init is
            True. Default equals to eps.
  :param src: the source examples. Default is None.
  :param sanity_checks: bool, if True, include asserts (Turn them off to use 
            less runtime memory or for unit tests that intentionally pass strange input)
  :return: a tensor of adversarial examples.
  """
  if norm == 1:
    raise NotImplementedError(
        "It's not clear that FGM is a good inner loop"
        " step for PGD when norm=1, because norm=1 FGM "
        " changes only one pixel at a time. We need "
        " to rigorously test a strong norm=1 PGD "
        "before enabling this feature."
    )
  if norm not in [np.inf, 2]:
    raise ValueError("Norm order must be either np.inf or 2.")
  if eps < 0:
    raise ValueError(
        "eps must be greater than or equal to 0, got {} instead".format(eps)
    )
  if eps == 0:
    return x
  if eps_iter < 0:
    raise ValueError(
        "eps_iter must be greater than or equal to 0, got {} instead".format(
            eps_iter
        )
    )
  if eps_iter == 0:
    return x

  assert eps_iter <= eps, (eps_iter, eps)
  if clip_min is not None and clip_max is not None:
    if clip_min > clip_max:
      raise ValueError(
          "clip_min must be less than or equal to clip_max, got clip_min={} and"
          " clip_max={}".format(clip_min, clip_max)
      )

  asserts = []

  # If a data range was specified, check that the input was in that range
  if clip_min is not None:
    assert_ge = torch.all(
        torch.ge(x, torch.tensor(clip_min, device=x.device, dtype=x.dtype))
    )
    asserts.append(assert_ge.detach().cpu())

  if clip_max is not None:
    assert_le = torch.all(
        torch.le(x, torch.tensor(clip_max, device=x.device, dtype=x.dtype))
    )
    asserts.append(assert_le.detach().cpu())

  # Initialize loop variables
  if rand_init:
    if rand_minmax is None:
      rand_minmax = eps
    eta = torch.zeros_like(x).uniform_(-rand_minmax, rand_minmax)
  else:
    eta = torch.zeros_like(x)

  eta = clip_eta(eta, norm, eps)
  adv_x = x + eta
  if clip_min is not None or clip_max is not None:
    adv_x = torch.clamp(adv_x, clip_min, clip_max)
  if y is None:
    # Using model predictions as ground truth to avoid label leaking.
    _, y = torch.max(model_classifier(model_featurizer(x)), 1)

  asserts.append(eps_iter <= eps)
  if norm == np.inf and clip_min is not None:
    asserts.append(eps + clip_min <= clip_max)

  if sanity_checks:
    assert np.all(asserts)

  i = 0
  while i < nb_iter:
    adv_x = fast_gradient_method(
        model_featurizer=model_featurizer,
        model_classifier=model_classifier,
        model_discriminator=model_discriminator,
        hparams=hparams,
        x=adv_x,
        eps=eps_iter,
        norm=norm,
        clip_min=clip_min,
        clip_max=clip_max,
        y=y,
        targeted=targeted,
        adv_type=adv_type,
        src=src,
        x_natural=x,
        sanity_checks=sanity_checks,
    )
    # Clipping perturbation eta to norm ball.
    eta = adv_x - x
    eta = clip_eta(eta, norm, eps)
    adv_x = x + eta
    # Redo the clipping.
    # FGM already did it, but subtracting and re-adding eta can add some
    # small numerical error.
    if clip_min is not None or clip_max is not None:
      adv_x = torch.clamp(adv_x, clip_min, clip_max)
    i += 1

  return adv_x
