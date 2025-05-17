from torch import nn


class Decoder(nn.Module):
    def __init__(self, output_size):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(128, 256)  # Example input size
        self.fc2 = nn.Linear(256, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
