# vq-ensemble

This is an experiment to see if a VQ-DRAW like model can be used to produce an ensemble of successful classifiers by generating random, well-performing weights.

The experiment is setup as follows. First, I use an MLP refinement network that takes in an entire (flattened) collection of NN weights, and outputs a new set of NN weight options. In the outer training loop, random latent codes are chosen, and their weights are decoded. These weights are then tuned with one step of SGD on a dataset, and then the decoded outputs from the refinement network are pulled closer to the post-SGD-step weights. Thus, at first the refinement network outputs random network weights, but over time it should learn to only output weights that perform well on the dataset.

The end goal is to produce model-generating-models that can generate random, well-performing classifiers (or other NNs). Since the model-generating-model is effectively a representation of an ensemble, it may perform better at the classification task. The hope would be that finding such a model would greatly reduce overfitting, and make better use of the data.
