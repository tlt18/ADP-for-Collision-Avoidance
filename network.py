import torch.nn as nn
import numpy as np


class Actor(nn.Module):
    def __init__(self, inputSize, Outputsize, lr):
        super().__init__()
        self._out_gain = PI / 9
        self._norm_matrix = 0.1 * \
            torch.tensor([1, 1, 1, 1], dtype=torch.float32)
        # NN
        self.layers = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, output_size),
            nn.Tanh()
        )
        # optimizer
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.opt, 100, gamma=0.9, last_epoch=-1)
        self._initializeWeights()
        # zeros state value
        self._zero_state = torch.tensor([0.0, 0.0, 0.0, 0.0])

    def forward(self, x):
        temp = torch.mul(x, self._norm_matrix)
        x = torch.mul(self._out_gain, self.layers(temp))
        return x

    def predict(self, x):
        return self.forward(x).detach().numpy()

    def saveParameters(self, x):
        pass

    def _initializeWeights(self):
        """
        initial parameter using xavier
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                init.constant_(m.bias, 0.0)


class Critic(nn.Module):
    def __init__(self, inputSize, Outputsize, lr):
        super().__init__()
        # initial parameters of actor
        self.layers = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, output_size),
            nn.ReLU()
        )
        self._norm_matrix = 0.1 * \
            torch.tensor([2, 5, 10, 10], dtype=torch.float32)
        # initial optimizer
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.opt, 100, gamma=0.9, last_epoch=-1)
        self._initializeWeights()
        # zeros state value
        self._zero_state = torch.tensor([0.0, 0.0, 0.0, 0.0])

    def forward(self, x):
        x = torch.mul(x, self._norm_matrix)
        x = self.layers(x)
        return x

    def predict(self, x):
        return self.forward(state).detach().numpy()

    def saveParameters(self, x):
        pass

    def _initializeWeights(self):
        """
        initial paramete using xavier
        """

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                init.constant_(m.bias, 0.0)
