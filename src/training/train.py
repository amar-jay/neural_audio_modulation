from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from models.encoder import Encoder
from models.decoder import Decoder
from data.dataset import AudioDataset
from utils.metrics import calculate_metrics
import yaml


def load_config(config_path="src/config/default.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def train_model(config):
    # Initialize dataset and dataloader
    train_dataset = AudioDataset(config["data"]["train_data_path"])
    train_loader = DataLoader(
        train_dataset, batch_size=config["training"]["batch_size"], shuffle=True
    )

    # Initialize model, loss function, and optimizer
    encoder = Encoder(config["model"])
    decoder = Decoder(config["model"])
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=config["training"]["learning_rate"],
    )

    # Training loop
    for epoch in range(config["training"]["num_epochs"]):
        encoder.train()
        decoder.train()
        for batch in train_loader:
            audio_input = batch["audio"]
            optimizer.zero_grad()

            # Forward pass
            encoded = encoder(audio_input)
            decoded = decoder(encoded)

            # Compute loss
            loss = criterion(decoded, audio_input)
            loss.backward()
            optimizer.step()

        # Evaluate model
        if (epoch + 1) % config["training"]["eval_interval"] == 0:
            evaluate_model(encoder, decoder, train_loader)


def evaluate_model(encoder, decoder, data_loader):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        for batch in data_loader:
            audio_input = batch["audio"]
            encoded = encoder(audio_input)
            decoded = decoder(encoded)

            # Calculate metrics
            metrics = calculate_metrics(audio_input, decoded)
            print(f"Evaluation Metrics: {metrics}")


if __name__ == "__main__":
    config = load_config()
    train_model(config)
