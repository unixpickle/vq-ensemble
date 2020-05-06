# vq-ensemble

This is an experiment to see if a VQ-DRAW like model can be used to produce an ensemble of successful classifiers by generating random, well-performing weights. This is a form of meta-learning.

# Premise

The experiment is setup as follows. First, I use an MLP refinement network that takes in an entire (flattened) collection of NN weights, and outputs a new set of NN weight options. In the outer training loop, random latent codes are chosen, and their weights are decoded. These weights are then tuned with one step of SGD on a dataset, and then the decoded outputs from the refinement network are pulled closer to the post-SGD-step weights. Thus, at first the refinement network outputs random network weights, but over time it should learn to only output weights that perform well on the dataset.

The end goal is to produce model-generating-models that can generate random, well-performing classifiers (or other NNs). Since the model-generating-model is effectively a representation of an ensemble, it may perform better at the classification task. The hope would be that finding such a model would greatly reduce overfitting, and make better use of the data.

# Hyper-parameters

There are a few HPs carried over from VQ-DRAW. The number of stages and the number of options define the size of the ensemble. My hypothesis is that larger ensembles will generally be better, but of course they will also require more memory.

The inner loop is implemented as vanilla SGD, so the inner loop step size is important to get right. The outer loop can be implemented with any optimizer, so it's easier to tune. Perhaps in the future the inner loop could be implemented with multiple SGD steps, making it possible to use momentum or even Adam.

The entire refinement network architecture is a hyperparameter. For now, I use a small MLP with different output biases for every option and stage. It should be verified that the MLP even helps. If not, the refinement network is just leaning on the output biases (likely in the first few stages) to produce good models, in which case the ensemble is rather small.

# Experiments

For the inner-loop model, I chose a small CNN with two strided convolutions followed by two fully-connected layers. The model uses ReLU activations. I did not employ dropout regularization for vq-ensemble, although I did use it for a baseline.

## Baseline

As a baseline, I directly trained the CNN with Adam (using an initial LR of 1e-3 and decaying it to 1e-4). This gives an idea of how much the model is capable of overfitting, and how well it does without any ensembling. This experiment can be run with [train_mnist_baseline.py](train_mnist_baseline.py).

```
...
step: 41999: train=0.000696 test=0.023440
Evaluation accuracy (train): 100.00%
Evaluation accuracy (test): 98.61%
```

## Adding dropout

In this experiment, I added dropout to the baseline (setting `DROPOUT = True` at the top of `train_mnist_baseline.py`). Since dropout creates an implicit ensemble, this is a good baseline for vq-ensemble.

```
RESULTS HERE
```

## vq-ensemble (take 1)

In this experiment, I trained a vq-ensemble model with 40 stages and 4 options (a total of 4^40 = 1.2 * 10^24 models). I trained the model with fixed hyperparameters for a few hours. I then evaluated an ensemble of 16 randomly sampled models, where models' outputs were added after the softmax.

```
RESULTS HERE
```

## vq-ensemble (ablation)

In this experiment, I replace the refinement network with a set of learnable biases. Each (stage, option) pair has a different, learned output. This removes the intelligence from the ensemble generation process, and drastically limits the distribution of model ensembles which can be learned.

This experiment can be run by setting `NO_NN = True` in the [train_mnist.py](train_mnist.py) script.

```
RESULTS HERE
```
