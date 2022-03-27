import copy
from random import random

import numpy as np
import torch as T
import torch.nn as nn
import torch
from . import settings


class DeepQNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, training_strategy_fn, logger, lr=0.00025):
        super(DeepQNetwork, self).__init__()
        self.input_dims = settings.INPUT_DIMENSIONS
        _, c, h, w = input_dims
        self.logger = logger
        self.n_actions = n_actions
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.burnin = settings.BURENING
        self.learn_every = settings.LEARN_EVERY
        self.sync_every = settings.SYNC_EVERY
        self.save_every = settings.SAVE_EVERY
        self.training_strategy = training_strategy_fn
        self.chkpt_dir = 'tmp/dueling_ddqn'
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

    def td_estimate(self, state, action):
        current_Q = self.Q_eval(state, "online")[
            torch.arange(0, settings.BATCH_SIZE, dtype=torch.long),
            action]
        return current_Q

    def update_online(self, td_estimate, td_target, optimizer):
        loss = self.loss_fn(td_estimate, td_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    def update_target(self):
        self.target.load_state_dict(self.online.state_dict())

    @torch.no_grad()
    def td_target(self, reward, next_state):
        next_state_Q = self.Q_eval(next_state, "online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.model(next_state, model="target")[
            torch.arange(0, settings.BATCH_SIZE, dtype=torch.long),
            best_action]
        return (reward + settings.GAMMA * next_Q).float()

    def racall(self, transitions):
        batch = random.sample(transitions, settings.BATCH_SIZE)

        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    """def epsilon_update(self):
        self.epsilon = self.min_epsilon if self.t >= self.decay_steps else self.epsilons[self.t]
        self.t += 1
        return self.epsilon"""
    def act(self, train, valid_actions, action_ideas, game_state):
        self.logger.debug(f"inside act {self.training_strategy}")
        action = self.training_strategy.select_action(train=train, valid_actions=valid_actions,
                                                       action_ideas=action_ideas,
                                                       game_state=game_state, Q_eval=self)
        return action

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))



