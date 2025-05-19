import argparse
import torchaudio
import torch
import torch.nn as nn
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from torchaudio import transforms as T
from ..models.model import NeuralAudioEncoding
from ..data.dataset import AudioDataset
from ..data.preprocessing import transform_audio
from ..utils.metrics import calculate_metrics
from ..utils.audio_utils import load_audio, save_audio, play_audio


def load_config(config_path="src/config/default.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def load_model(model_path, config):
    """Load the trained model"""
    model = NeuralAudioEncoding(
        input_dim=config["model"]["input_dim"],
        num_layers=config["model"]["num_layers"],
        step_dim=int(config["model"]["compression_ratio"] ** -1),
        use_positional_encoding=config["model"]["use_positional_encoding"],
        positional_multires=config["model"]["positional_multires"],
        no_channels=config["model"]["no_channels"],
    )

    try:
        # Try loading with standard method
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    except (IndexError, RuntimeError) as e:
        print(f"Standard loading failed: {e}")
        print("Trying alternative loading method...")
        try:
            # Try loading entire model instead of just state dict
            loaded_model = torch.load(model_path, map_location=torch.device("cpu"))
            if isinstance(loaded_model, NeuralAudioEncoding):
                model = loaded_model
            elif hasattr(loaded_model, "state_dict"):
                model.load_state_dict(loaded_model.state_dict())
            else:
                print(
                    "Warning: Could not determine how to load the model. Using uninitialized model."
                )
        except Exception as e2:
            print(f"Alternative loading also failed: {e2}")
            print("Warning: Using uninitialized model weights.")

    print()
    print(model)
    print()
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Model size: {model.storage_size()} MB")

    return model


def test_single_file(
    model, file_path, config, device, output_dir=None, visualize=False, play=False
):
    """Run inference on a single audio file"""
    print(f"Processing file: {file_path}")

    # Load and preprocess audio
    audio_tensor, sample_rate = torchaudio.load(file_path)
    print(f"Original audio shape: {audio_tensor.shape}, Sample rate: {sample_rate}")
    audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension

    # Apply transformations if specified
    if config["data"].get("transform"):
        transform = transform_audio(**config["data"]["transform"])
        audio_tensor = transform(audio_tensor, sample_rate)

    # Add channel dimension if needed
    if len(audio_tensor.shape) == 2:
        audio_tensor = audio_tensor.unsqueeze(-1)

    audio_tensor = audio_tensor.to(device)

    # Run inference
    model.eval()
    with torch.no_grad():
        audio_output, embedding_loss, perplexity, embeds = model(audio_tensor)
        print(f"Input shape: {audio_tensor.shape}")
        print(f"Output shape: {audio_output.shape}")
        print(f"Compressed Embedding shape: {embeds.shape}")
        print(f"Compression ratio per number of parameters: {embeds.numel() / audio_tensor.numel()}")

        # Calculate metrics
        metrics = calculate_metrics(
            audio_tensor.detach().cpu(),
            audio_output.detach().cpu(),
            perplexity=perplexity.detach().cpu(),
            embedding_loss=embedding_loss.detach().cpu(),
            loss=torch.tensor(0.0),  # Placeholder
            config=config,
        )

        # Print metrics
        print("\nMetrics:")
        max_key_length = max(len(key) for key in metrics.keys())
        print("-" * (max_key_length + 15))  # Separator line
        print(f"{'Metric Name':<{max_key_length}}  |  {'Value'}")
        print("-" * (max_key_length + 15))  # Separator line

        for key, value in metrics.items():
            # Format value based on type
            if isinstance(value, float):
                formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)
            print(f"{key:<{max_key_length}}  |  {formatted_value}")
        print("-" * (max_key_length + 15))  # Separator line

        if play:
            audio_output = T.Resample(
                orig_freq=config["data"]["transform"]["sample_rate"], new_freq=sample_rate
            )(audio_output.detach().cpu())

            audio_tensor = T.Resample(
                orig_freq=config["data"]["transform"]["sample_rate"], new_freq=sample_rate
            )(audio_tensor.detach().cpu())
            # Play original and processed audio
            print("Playing original audio...")
            play_audio(audio_tensor[0].detach().cpu(), sample_rate)
            print("Playing processed audio...")
            play_audio(audio_output[0].detach().cpu(), sample_rate)

        # Save output if requested
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_filename = os.path.join(output_dir, os.path.basename(file_path))
            save_audio(output_filename, audio_output[0].detach().cpu().numpy(), sample_rate)
            print(f"Saved processed audio to: {output_filename}")

        # Visualize if requested
        if visualize:
            plt.figure(figsize=(10, 6))
            plt.subplot(3, 1, 1)
            plt.plot(audio_tensor[0].detach().cpu().numpy(), label="Original")
            plt.title("Original Audio")
            plt.subplot(3, 1, 2)
            plt.plot(audio_output[0].detach().cpu().numpy(), label="Processed")
            plt.title("Processed Audio")

    return metrics


def test_dataset(model, config, device, output_dir=None, max_samples=None):
    """Run inference on the entire dataset and calculate average metrics"""
    # Initialize dataset and dataloader
    test_dataset = AudioDataset(config["data"]["dataset_path"])
    if config["data"].get("transform"):
        test_dataset.transform = transform_audio(**config["data"]["transform"])

    test_loader = DataLoader(
        test_dataset, batch_size=config["training"]["batch_size"], shuffle=False
    )

    # Limit number of samples if specified
    num_batches = len(test_loader)
    if max_samples is not None:
        num_batches = min(num_batches, max_samples // config["training"]["batch_size"] + 1)

    # Storage for aggregated metrics
    all_metrics = {}
    processed_samples = 0

    # Process dataset
    model.eval()
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(test_loader), total=num_batches, desc="Testing dataset"):
            if max_samples is not None and processed_samples >= max_samples:
                break

            # Get audio input and prepare
            audio_input = batch[0].to(device)

            # Add channel dimension if needed
            if len(audio_input.shape) == 2:
                audio_input = audio_input.unsqueeze(-1)

            # Forward pass
            audio_output, embedding_loss, perplexity, _ = model(audio_input)

            # Calculate metrics
            batch_metrics = calculate_metrics(
                audio_input.detach().cpu(),
                audio_output.detach().cpu(),
                perplexity=perplexity.detach().cpu(),
                embedding_loss=embedding_loss.detach().cpu(),
                loss=torch.tensor(0.0),  # Placeholder
                config=config,
            )

            # Aggregate metrics
            for key, value in batch_metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(value)

            # Save samples if requested
            if output_dir:
                for i in range(audio_output.shape[0]):
                    sample_idx = idx * config["training"]["batch_size"] + i
                    output_filename = os.path.join(output_dir, f"sample_{sample_idx}.wav")
                    save_audio(audio_output[i].detach().cpu().numpy(), output_filename)

            processed_samples += audio_input.shape[0]

            if idx % 10 == 0:
                print(f"Processed {processed_samples} samples...")

    # Calculate average metrics
    avg_metrics = {key: np.array(values).mean() for key, values in all_metrics.items()}

    # Print results
    print("\nAverage Metrics:")
    for key, value in avg_metrics.items():
        print(f"{key}: {value}")

    return avg_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Test Neural Audio Modulation model on dataset or single file"
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model")
    parser.add_argument("--file", type=str, help="Path to a single audio file for testing")
    parser.add_argument("--output_dir", type=str, help="Directory to save processed audio files")
    parser.add_argument(
        "--max_samples", type=int, help="Maximum number of samples to process from dataset"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Visualize input and output for single file mode"
    )
    parser.add_argument("--play", action="store_true", help="Play audio for single file mode")

    args = parser.parse_args()

    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load configuration
    config = load_config()

    # Load model
    model = load_model(args.model_path, config)
    model.to(device)

    # Create output directory if specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    # Run inference
    if args.file:
        # Single file mode
        test_single_file(
            model, args.file, config, device, args.output_dir, args.visualize, args.play
        )
    else:
        # Dataset mode
        test_dataset(model, config, device, args.output_dir, args.max_samples)


if __name__ == "__main__":
    main()
