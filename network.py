import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, inputSize, Outputsize, lr):
        super().__init__()

    def forward(self, x):
        pass

    def predict(self, x):
        pass

    def saveParameters(self, x):
        pass


class Critic(nn.Module):
    def __init__(self, inputSize, Outputsize, lr):
        super().__init__()

    def forward(self, x):
        pass

    def predict(self, x):
        pass

    def saveParameters(self, x):
        pass
