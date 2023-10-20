*This is not an officially supported Google product.*

# Welcome to DomainRobust

DomainRobust is a PyTorch testbed for evaluating the adversarial robustness of
domain adaptation algorithms. Specifically, we consider a unsupervised domain
adaptation (UDA) setting: given a labeled dataset from a source domain and
an unlabeled dataset from a (related) target domain, the goal is to train a
classifier that is robust against adversarial attacks on the target domain.
The implementation builds on [DomainBed](https://github.com/facebookresearch/DomainBed), [AutoAttack](https://github.com/fra31/auto-attack), and [CleverHans](https://github.com/cleverhans-lab/cleverhans).

## Available algorithms

The following [algorithms](domainrobust/algorithms.py) are currently available:

* Empirical Risk Minimization (ERM): standard ERM without adversarial training.
* Domain Adversarial Neural Network (DANN, [Ganin et al., 2015](https://arxiv.org/abs/1505.07818)): standard DANN without adversarial training.
* Adversarial Training (AT, [Madry et al., 2017](https://arxiv.org/abs/1706.06083)). Three variants are supported: (i) AT only on the labeled source data, (ii) AT on pseudo-labeled target data (where pseudo labels are obtained a-priori using DANN), and (iii) AT on the source data along with a DANN regularizer. See our paper for details.
* TRadeoff-inspired Adversarial DEfense via Surrogate-loss minimization (TRADES, [Zhang et al., 2019](https://arxiv.org/abs/1901.08573)). Two variants are supported: (i) TRADES only on the labeled source data, and (ii) TRADES on pseudo-labeled target data (where pseudo labels are obtained a-priori using DANN). See our paper for details.
* Adversarially Robust Training method for UDA (ARTUDA, [Lo, S.Y. and Patel, V., 2022](https://arxiv.org/abs/2202.09300))
* Meta self-training for robust unsupervised domain adaptation (SRoUDA, [Zhu et al., 2022](https://arxiv.org/abs/2212.05917))
* Divergence Aware adveRsarial Training (DART, in submission)

## Available datasets

DomainRobust includes the following datasets:

* DIGIT-FIVE (MNIST, MNIST-M, SVHN, SYN, USPS)
[Download](https://github.com/VisionLearningGroup/VisionLearningGroup.github.io/tree/master/M3SDA/code_MSDA_digit#digit-five-download)
* PACS ([Li et al., 2017](https://arxiv.org/abs/1710.03077))
* Office-Home ([Venkateswara et al., 2017](https://arxiv.org/abs/1706.07522))
* VISDA-17 ([Overview](https://ai.bu.edu/visda-2017/))

## Quick start

To train a single model:

```sh
python3 -m scripts.train\
       --data_dir=/my/datasets/path\
       --algorithm AT\
       --dataset DIGIT\
       --task domain_adaptation\
       --source_envs 0\
       --target_envs 2\
       --eps 0.008\
       --atk_lr 0.004\
       --atk_iter 5\
       --attack pgd\
       --source_type clean\
       --pretrain_model_dir=/my/pretrained/model/path \
       --pretrain
```

To launch a sweep (over a range of hyperparameters and possibly multiple algorithms and datasets):

```sh
python -m scripts.sweep launch\
       --data_dir=/my/datasets/path\
       --output_dir=/my/sweep/output/path\
       --command_launcher MyLauncher\
       --algorithms AT TRADES SROUDA\
       --datasets DIGIT\
       --n_hparams 20\
       --n_trials 3\
       --task domain_adaptation\
       --source_envs 0\
       --target_envs 1 2 3 4\
       --eps 0.008\
       --atk_lr 0.004\
       --atk_iter 5\
       --attack pgd\
       --source_type clean\
       --pretrain_model_dir=/my/pretrained/model/path \
       --pretrain
```


Here, `MyLauncher` has three options as implemented in `command_launchers.py`: local, dummy, multi_gpu. The command above trains multiple models for a number of randomly sampled hyperparameter sets (specified using `n_hparams`). For each model (defined by a particular choice of an algorithm, a dataset, and a hyperparameter set), an output directory is automatically created. When the training process is complete, an empty file named Done is created in the directory. Moreover, the directory will be populated with checkpoints of the best models based on clean/robust source/target validation set accuracy, in addition to a training log.

Once all jobs have reached either a successful or failed state, you can proceed to remove the records associated with the failed jobs using the command ``python -m domainrobust.scripts.sweep delete_incomplete``, so that the folders associated with the incomplete jobs can be deleted, otherwise other sweep will not launch the job with the same hyperparameters. After deleting the incomplete jobs, you can re-launch them by executing ``python -m domainrobust.scripts.sweep launch``. Please ensure that you provide the same command-line arguments when you re-launch as you did during the initial launch.

To collect the results from all folders:

````sh
python -m domainrobust.scripts.collect_results\
       --latex\
       --input_dir=/my/sweep/output/path\
       --task domain_adaptation\
       --attack pgd
````

To evaluate on existing models:
````sh
python -m domainrobust.scripts.test\
       --data_dir=/my/datasets/path \
       --input_dir=/gcs/xcloud-shared/${USER}/output \
       --dataset DIGIT \
       --eps 0.008 \
       --atk_lr 0.004 \
       --atk_iter 20 \
       --attack pgd 
````