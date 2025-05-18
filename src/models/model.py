from torch import nn
import torch

import torch
import torch.nn as nn
from .encoding import PositionalEncoder
from .quantization import VectorQuantizer
import torch.nn.functional as F


class NeuralAudioEncoding(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_layers: int = 3,
        step_dim: int = 2,
        use_positional_encoding: bool = True,
        positional_multires: int = 1,
        no_channels: int = 10,
    ):
        super(NeuralAudioEncoding, self).__init__()

        self.input_dim = input_dim
        self.num_layers = num_layers
        self.step_dim = step_dim
        self.use_positional_encoding = use_positional_encoding
        # Add positional encoding if requested
        if use_positional_encoding:
            self.positional_encoder = PositionalEncoder(
                multires=positional_multires, no_channels=no_channels
            )
            # Adjust input dimension to account for the positional encoding
            augmented_input_dim = self.positional_encoder.out_dim
        else:
            augmented_input_dim = input_dim

        if augmented_input_dim != input_dim:
            print(
                f"Input dimension changed from {input_dim} to {augmented_input_dim} due to positional encoding."
            )
            self.proj_input = nn.Linear(augmented_input_dim, input_dim)

        # Calculate encoder dimensions
        encoder_dims = [input_dim]
        for i in range(num_layers):
            encoder_dims.append(encoder_dims[-1] // step_dim)

        # Calculate decoder dimensions
        decoder_dims = encoder_dims.copy()
        decoder_dims.reverse()  # So we go from smallest to largest
        print(f"encoder_dims: {encoder_dims}, decoder_dims: {decoder_dims}")

        # Create encoder layers
        self.encoder_layers = nn.ModuleList(
            [nn.Linear(encoder_dims[i], encoder_dims[i + 1]) for i in range(num_layers)]
        )

        # Create decoder layers with skip connections
        self.decoder_layers = nn.ModuleList(
            [nn.Linear(decoder_dims[i], decoder_dims[i + 1]) for i in range(num_layers)]
        )

        # layer normalizations
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(encoder_dims[i+1]) for i in range(num_layers)] +
            [nn.LayerNorm(decoder_dims[i+1]) for i in range(num_layers)]
        )

        self.proj_out = nn.Linear(decoder_dims[-1], 1)

        # Vector quantization bottleneck
        notebook_dim = input_dim // (step_dim * num_layers)
        print(f"Vector quantization bottleneck dimension: {notebook_dim}")
        self.bottleneck = VectorQuantizer(n_e=notebook_dim, e_dim=notebook_dim, beta=0.25)

        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with skip connections and optional positional encoding."""
        # Apply positional encoding if enabled
        if self.use_positional_encoding:
            x = self.positional_encoder(x.unsqueeze(-1))

        if hasattr(self, "proj_input"):
            x = self.proj_input(x)
        # Store encoder outputs for skip connections
        encoder_outputs = []

        # Encoder forward pass
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x)
            encoder_outputs.append(x)
            x = self.layer_norms[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=0.5, training=True)  # randomly zero out 50% of activations

        z_e = x
        # Apply the vector quantization bottleneck
        embedding_loss, x, perplexity, _, _ = self.bottleneck(x)

        n = len(self.encoder_layers)
        # Decoder forward pass with skip connections
        for i, layer in enumerate(self.decoder_layers):
            # Add skip connection - connect with corresponding encoder output
            skip_idx = self.num_layers - 1 - i  # Index from the end
            x = x + encoder_outputs[skip_idx]  # Skip connection
            x = layer(x)
            if i < self.num_layers - 1:  # No activation after final layer
                x = self.layer_norms[n+i](x)
                x = self.activation(x)
                x = F.dropout(x, p=0.5, training=True)  # randomly zero out 50% of activations

        if hasattr(self, "proj_out"):
            x = self.proj_out(x)
        # remove the last dimension
        return x.squeeze(-1), embedding_loss, perplexity, z_e

    def storage_size(self) -> int:
        """Calculate the storage size of the model in MB"""
        # Calculate the number of parameters in the model
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # Convert to MB
        storage_size_mb = num_params * 4 / (1024 ** 2)
        return storage_size_mb

    def loss(self, x, x_hat, embedding_loss) -> torch.Tensor:
        x_train_var = torch.var(x, dim=0, unbiased=False).mean()
        recon_loss = torch.mean((x_hat - x) ** 2) / x_train_var
        loss = recon_loss + embedding_loss
        return loss

    def encoded(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_positional_encoding:
            x = self.positional_encoder(x.unsqueeze(-1))

        if hasattr(self, "proj_input"):
            x = self.proj_input(x)
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x)
            x = self.layer_norms[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=0.5, training=True)  # randomly zero out 50% of activations

        return x

    def decoded(self, x: torch.Tensor, encoder_outputs=None) -> torch.Tensor:
        if encoder_outputs is None:
            # Without skip connections
            n = len(self.decoder_layers)
            for i, layer in enumerate(self.decoder_layers):
                x = layer(x)
                if i < self.num_layers - 1:
                    x = self.layer_norms[n+i](x)
                    x = self.activation(x)
                    x = F.dropout(x, p=0.5, training=True)  # randomly zero out 50% of activations
        else:
            n = len(self.encoder_outputs)
            # With provided skip connections
            for i, layer in enumerate(self.decoder_layers):
                skip_idx = self.num_layers - 1 - i
                x = torch.cat([x, encoder_outputs[skip_idx]], dim=-1)
                x = layer(x)
                if i < self.num_layers - 1:
                    x = self.layer_norms[n+i](x)
                    x = self.activation(x)
                    x = F.dropout(x, p=0.5, training=True)  # randomly zero out 50% of activations
        if hasattr(self, "proj_out"):
            x = self.proj_out(x)

        return x
