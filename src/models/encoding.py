import torch
import torch.nn as nn


class PositionalEncoder(nn.Module):
    def __init__(self, multires, no_channels):
        super().__init__()
        self.multires = multires
        self.input_dims = no_channels
        self.freq_bands = 2.0 ** torch.linspace(0.0, multires - 1, steps=multires)
        self.out_dim = self.input_dims + self.input_dims * multires * 2
        self.register_buffer("freq_bands_view", self.freq_bands.view(1, 1, -1))

    def forward(self, inputs):
        orig_shape = inputs.shape
        inputs_flat = inputs.reshape(-1, self.input_dims)
        embeddings = [inputs_flat]
        x_expanded = inputs_flat.unsqueeze(-1)
        freqs = x_expanded * self.freq_bands_view
        embeddings.append(torch.sin(freqs).reshape(inputs_flat.shape[0], -1))
        embeddings.append(torch.cos(freqs).reshape(inputs_flat.shape[0], -1))
        embedded = torch.cat(embeddings, dim=-1)
        embedded = embedded.reshape(*orig_shape[:-1], self.out_dim)
        return embedded