import copy

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims,
                 n_actions, that):
        super(DeepQNetwork, self).__init__()
        self.input_dims = 3*17*17
        _, c, h, w = input_dims
        self.that = that
        self.n_actions = n_actions

        if h != 17:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 17:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1600, 1600),
            nn.ReLU(),
            nn.Linear(1600, 512),
            nn.ReLU(),
            nn.Linear(512, self.n_actions),
        )
        self.target = copy.deepcopy(self.online)

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)