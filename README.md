# vq-ensemble

This is a one-day experiment to see if a VQ-DRAW-like model can be used to produce an ensemble of successful classifiers by generating random, well-performing weights.

**Result summary:** while this technique does seem to provide a way to train an ensemble of models, the ensemble does not perform much better than any one model. Furthermore, a model trained with dropout outperforms a model trained with vq-ensemble.

# Premise

The experiment is setup as follows. First, there is an inner-loop model that we'd like to learn an ensemble for. This model, `f(theta, x)`, takes in a set of parameters and an input and produces some prediction. We define a meta refinement network, `R(stage, theta)`, which produces a set of `K` different parameter vectors `theta'_1` through `theta'_K` based on parameters `theta`. We can sample random parameters by starting with `theta = zeros`, feeding the it into `R`, choosing a random output option, and repeating multiple times. These random parameters are all functions of `R`'s own parameters, which we don't explicitly denote.

In the meta-training loop, we repeatedly sample random a `theta`, take a random batch of inner-loop data `(x, y)`, perform one SGD step on `f(theta, x)`, and then pull the random `theta` closer to the post-SGD-step weights. This way, we gradually pull all of the outputs of `R` closer to trained models.

The end goal is to produce model-generating-models that can generate random but well-performing NNs. Since the model-generating-model is effectively a representation of an ensemble, it may perform better at the classification task. The hope would be that finding such a model would greatly reduce overfitting, and make better use of the available training data.

# Hyper-parameters

There are a few HPs carried over from VQ-DRAW. The number of stages and the number of options define the size of the ensemble. My hypothesis is that larger ensembles will generally be better, but of course they will also require more memory.

The inner loop is implemented as vanilla SGD, so the inner loop step size is important to get right. The outer loop can be implemented with any optimizer, so it's easier to tune. Perhaps in the future the inner loop could be implemented with multiple SGD steps, making it possible to use momentum or even Adam.

The entire refinement network architecture is a hyperparameter. For now, I use a small MLP with different output biases for every option and stage. It should be verified that the MLP even helps. If not, the refinement network is just leaning on the output biases (likely in the first few stages) to produce good models, in which case the ensemble is rather small.

# Experiments

For the inner-loop model, I chose a small CNN with two strided convolutions followed by two fully-connected layers. The model uses ReLU activations. The model has a total of 60106 parameters. I did not employ dropout regularization for vq-ensemble, although I did use it for a baseline.

All of these experiments were run on the MNIST dataset. Why MNIST? Because it's small and easy to iterate on. When ideas work, I scale them to harder datasets, but MNIST is usually a sane starting point.

## Baseline

As a baseline, I directly trained the CNN with Adam (using an initial LR of 1e-3 and decaying it to 1e-4). This gives an idea of how much the model is capable of overfitting, and how well it does without any ensembling.

This experiment can be run with [train_mnist_baseline.py](train_mnist_baseline.py).

```
...
Evaluation accuracy (train): 100.00%
Evaluation accuracy (test): 98.61%
```

## Baseline with Dropout

In this experiment, I added dropout to the baseline. Since dropout creates an implicit ensemble, this is a good baseline for vq-ensemble.

To run this experiment, set `DROPOUT = True` in [train_mnist_baseline.py](train_mnist_baseline.py).

```
Evaluation accuracy (train): 99.81%
Evaluation accuracy (test): 99.04%
```

## vq-ensemble (default)

In this experiment, I trained a vq-ensemble model with 40 stages and 4 options (a total of 4^40 = 1.2 * 10^24 models). I trained the model with fixed hyperparameters for a few hours (6300 iterations). I then evaluated an ensemble of 16 randomly sampled models, where models' outputs were added after the softmax.

This experiment can be run with [train_mnist.py](train_mnist.py).

```
Evaluation accuracy (train): 99.92% (single 99.88%)
Evaluation accuracy (test): 98.82% (single 98.76%)
Output variance: 0.134600
```

## vq-ensemble (ablation)

In this experiment, I replaced the refinement network with a set of learnable biases. Thus, each `(stage, option)` pair has a single learned output. This removes the intelligence from the ensemble generation process, and drastically limits the distribution of model ensembles which can be learned.

The purpose of this experiment is to see how important the refinement network actually is in learning a usable ensemble. If the ablation works as well as the original, then the refinement network is doing nothing. Otherwise, the refinement network is at least adding *something* useful to the manifold of possible ensembles.

This experiment can be run by setting `NO_NN = True` in [train_mnist.py](train_mnist.py).

```
Evaluation accuracy (train): 87.95% (single 79.92%)
Evaluation accuracy (test): 87.71% (single 80.45%)
Output variance: 0.445067
```

## vq-ensemble (extra large)

In this experiment, I tried tacking on an extra two layers to the refinement network, and I also upped the number of sampled models in the evaluation ensemble. Basically, this was the last push to get better results with the technique.

This experiment can be run by setting `EXTRA_LARGE = True` and `EVAL_ENSEMBLE = 48` in [train_mnist.py](train_mnist.py).

```
RESULTS HERE
```
