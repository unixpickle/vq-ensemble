import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class Encoder(nn.Module):
    def __init__(self, refiner):
        super().__init__()
        self.refiner = refiner

    def forward(self, targets):
        """
        Encode and decode the targets to produce a
        reconstruction sequence.

        Args:
            targets: target parameters, [N x param_size].

        Returns:
            A batch of reconstruction sequences, of shape
              [N x T x param_size].
        """
        batch = targets.shape[0]
        current_outputs = torch.zeros_like(targets)
        results = []
        for i in range(self.refiner.num_stages):
            outs = self._run_refiner(i, current_outputs)
            losses = torch.mean(torch.pow(outs - targets[:, None], 2), dim=-1)
            indices = torch.argmin(losses, dim=1)
            current_outputs = outs[range(batch), indices]
            results.append(current_outputs)
        return torch.stack(results, dim=1)

    def random_latents(self, batch):
        """
        Sample a batch of random latents.

        Args:
            batch: the batch size.

        Returns:
            An [N x T] latent tensor.
        """
        return torch.randint(self.refiner.num_options, (batch, self.refiner.num_stages))

    def decode(self, latents):
        """
        Decode a batch of random latents.

        Args:
            latents: an [N x T] tensor.

        Returns:
            A batch of reconstruction sequences, of shape
              [N x T x param_size].
        """
        batch = latents.shape[0]
        device = next(self.parameters()).device
        current_outputs = torch.zeros(batch, self.refiner.param_size, device=device)
        results = []
        for i in range(self.refiner.num_stages):
            outs = self._run_refiner(i, current_outputs)
            indices = latents[:, i]
            current_outputs = outs[range(batch), indices]
            results.append(current_outputs)
        return torch.stack(results, dim=1)

    def _run_refiner(self, stage, inputs):
        inputs = inputs.requires_grad_(True)
        return checkpoint(lambda x: self.refiner.forward(stage, x), inputs)


class Refiner(nn.Module):
    def __init__(self, param_size, num_stages, num_options=4, hidden_size=512, no_nn=False):
        super().__init__()
        self.param_size = param_size
        self.num_stages = num_stages
        self.num_options = num_options
        self.no_nn = no_nn

        self.stage_embedding = nn.Parameter(torch.randn(num_stages, hidden_size))
        self.biases = nn.Parameter(torch.randn(num_stages, num_options, param_size))
        self.output_scales = nn.Parameter(0.1 * torch.ones(num_stages))
        self.in_layer = nn.Sequential(
            nn.Linear(param_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.mid_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.out_layer = nn.Sequential(
            nn.Linear(hidden_size, param_size * num_options),
        )

    def forward(self, stage, params_in):
        """
        Apply one step of the refinement network.

        Args:
            stage: the stage index.
            params_in: the parameter batch, [N x param_size].

        Returns:
            A batch of refinements, [N x K x param_size].
        """
        if self.no_nn:
            out = self.biases[stage]
        else:
            embedding = self.stage_embedding[stage]
            out = self.in_layer(params_in)
            out = out + embedding
            out = self.mid_layer(out)
            out = out + embedding
            out = self.out_layer(out)
            out = out.view(params_in.shape[0], self.num_options, self.param_size)
            out = out + self.biases[stage]
        return out * self.output_scales[stage] + params_in[:, None]
