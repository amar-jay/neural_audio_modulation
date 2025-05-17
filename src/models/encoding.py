import torch
import torch.nn as nn
class PositionalEncoder(nn.Module):
    def __init__(self, multires, no_channels):
        """
        Positional encoder using multiresolution sinusoidal functions.

        Args:
            multires: number of frequency bands to use
            no_channels: number of audio channels (e.g., 1 for mono, 2 for stereo)
        """
        super().__init__()
        self.multires = multires

        # Fixed parameters
        self.input_dims = no_channels

        # Pre-compute frequency bands with log sampling
        self.freq_bands = 2.0 ** torch.linspace(0.0, multires - 1, steps=multires)

        # Calculate output dimension (input + sin + cos for each frequency band)
        self.out_dim = (
            self.input_dims + self.input_dims * multires * 2
        )  # input + sin/cos per dimension per frequency

        # Register frequency bands as a buffer
        self.register_buffer("freq_bands_view", self.freq_bands.view(1, 1, -1))

    def forward(self, inputs):
        orig_shape = inputs.shape
        inputs_flat = inputs.reshape(-1, self.input_dims)

        # List to store embeddings
        embeddings = [inputs_flat]  # Always include raw inputs

        # Calculate all frequencies at once: [batch_size, 3, num_freqs]
        x_expanded = inputs_flat.unsqueeze(-1)
        freqs = x_expanded * self.freq_bands_view

        # Apply sin and cos and add to embeddings
        embeddings.append(torch.sin(freqs).reshape(inputs_flat.shape[0], -1))
        embeddings.append(torch.cos(freqs).reshape(inputs_flat.shape[0], -1))

        # Concatenate all embeddings
        embedded = torch.cat(embeddings, dim=-1)

        # Restore original batch dimensions
        # print(
        #     f"\n\norig_shape: {orig_shape}, out shape: {self.out_dim}, embedded shape: {embedded.shape}"
        # )
        embedded = embedded.reshape(*orig_shape[:-1], self.out_dim)
        # print(f"embedded shape after reshape: {embedded.shape}")

        return embedded
