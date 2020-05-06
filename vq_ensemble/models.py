import math

import numpy as np
import torch
import torch.nn as nn


class Model(nn.Module):
    def parameter_scales(self):
        """
        Return, for each parameter in self.parameters(), a
        scale which should be applied to Gaussian noise to
        make it behave well as a parameter value.

        This is essentially an initializer scale.
        """
        results = []
        for p in self.parameters():
            if len(p.shape) == 4:
                results.append(1 / math.sqrt(p.shape[1] * p.shape[2] * p.shape[3]))
            elif len(p.shape) == 2:
                results.append(1 / math.sqrt(p.shape[1]))
            elif len(p.shape) == 1:
                results.append(1 / math.sqrt(p.shape[0]))
            else:
                raise ValueError('unable to deal with shape: ' + str(p.shape))
        return results

    def param_size(self):
        """
        Get the total number of parameters.
        """
        return sum(int(np.prod(x.shape)) for x in self.parameters())

    def set_parameters(self, flattened):
        flattened = flattened.detach().contiguous()
        scales = self.parameter_scales()
        idx = 0
        for scale, param in zip(scales, self.parameters()):
            size = int(np.prod(param.shape))
            sub_data = flattened[idx:idx+size]
            idx += size
            param.detach().copy_(sub_data.view(param.shape) * scale)

    def get_parameters(self):
        results = []
        scales = self.parameter_scales()
        for scale, param in zip(scales, self.parameters()):
            results.append(param.detach().contiguous().view(-1) / scale)
        return torch.cat(results, dim=0)


class MNISTModel(Model):
    """
    MNISTModel is a convolutional classifier for MNIST.
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(7*7*32, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )

    def forward(self, x):
        return self.fc(self.conv(x).view(x.shape[0], -1))
