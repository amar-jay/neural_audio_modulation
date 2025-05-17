from torch import nn


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, output, target):
        # Mean Squared Error Loss
        mse_loss = nn.MSELoss()(output, target)
        # Additional loss components can be added here
        return mse_loss


def calculate_loss(output, target):
    loss_fn = CustomLoss()
    return loss_fn(output, target)
