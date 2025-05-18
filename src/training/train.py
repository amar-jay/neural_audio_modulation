from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from ..models.model import NeuralAudioEncoding
from ..data.dataset import AudioDataset
from ..data.preprocessing import transform_audio
from ..utils.metrics import calculate_metrics
from ..utils.audio_utils import save_model
import yaml
import os
import wandb
from tqdm import tqdm


is_cuda_available = torch.cuda.is_available()
device = "cuda" if is_cuda_available else "cpu"
print(f"Using device: {device}")


def load_config(config_path="src/config/default.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def train_model(config):
    logger = wandb.init(project="neural_audio_modulation", config=config)
    # Initialize dataset and dataloader
    train_dataset = AudioDataset(config["data"]["dataset_path"])
    if config["data"]["transform"]:
        train_dataset.transform = transform_audio(
            config["data"]["transform"]["min_max"], config["data"]["transform"]["softmax"]
        )

    train_loader = DataLoader(
        train_dataset, batch_size=config["training"]["batch_size"], shuffle=True
    )

    # Initialize model, loss function, and optimizer
    model = NeuralAudioEncoding(
        input_dim=config["model"]["input_dim"],
        num_layers=config["model"]["num_layers"],
        step_dim=int(config["model"]["compression_ratio"] ** -1),
        use_positional_encoding=config["model"]["use_positional_encoding"],
        positional_multires=config["model"]["positional_multires"],
        no_channels=config["model"]["no_channels"],
    )
    optimizer = optim.Adam(
        list(model.parameters()),
        lr=config["training"]["learning_rate"],
    )

    model.to(device)
    # Training loop
    for epoch in range(config["training"]["num_epochs"]):
        model.train()
        for idx, batch in tqdm(enumerate(train_loader), desc=f"Epoch {epoch + 1}"):
            # Properly reshape the audio input to have a channel dimension
            audio_input = batch[0].to(device)

            # Add channel dimension if needed
            if len(audio_input.shape) == 2:  # [batch_size, sequence_length]
                audio_input = audio_input.unsqueeze(-1)  # [batch_size, sequence_length, 1]

            optimizer.zero_grad()

            # Forward pass
            audio_output, embedding_loss, perplexity, _ = model(audio_input)

            # Compute loss
            loss = model.loss(audio_output, audio_input, embedding_loss)
            loss.backward()
            optimizer.step()

            # Evaluate model
            if (idx + 1) % config["training"]["eval_interval"] == 0:
                with torch.no_grad():
                    metrics = calculate_metrics(
                        audio_input.detach().cpu(),
                        audio_output.detach().cpu(),
                        perplexity=perplexity.detach().cpu(),
                        embedding_loss=embedding_loss.detach().cpu(),
                        loss=loss.detach().cpu(),
                        config=config,
                    )
                    wandb.log(metrics)
    logger.finish()
    save_model(model)



if __name__ == "__main__":
    import wandb

    import os

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    config = load_config()
    train_model(config)
