import random

import numpy as np
import torch

from . import settings

class EGreedyExpStrategy():
    def __init__(self, init_epsilon=1.0, min_epsilon=0.1, decay_steps=20000, preprocessor=None, logger =None):
        self.epsilon = init_epsilon
        self.init_epsilon = init_epsilon
        self.decay_steps = decay_steps
        self.min_epsilon = min_epsilon
        self.epsilons = 0.01 / np.logspace(-2, 0, decay_steps, endpoint=False) - 0.01
        self.epsilons = self.epsilons * (init_epsilon - min_epsilon) + min_epsilon
        self.t = 0
        self.exploratory_action_taken = None
        self.preprocessor = preprocessor
        self.logger = logger
        # test
        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1

    def _epsilon_update(self):
        self.epsilon = self.min_epsilon if self.t >= self.decay_steps else self.epsilons[self.t]
        self.t += 1
        return self.epsilon

    def _epsilon_update_str_2(self):
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)


    def select_action(self, train, valid_actions, action_ideas, game_state, Q_eval):
        if train and np.random.rand()  < self.epsilon:
            self.logger.debug(f"self.epsilon {self.epsilon}")
            self.logger.debug("Choosing action purely at random.")
            # 80%: walk in any direction. 10% wait. 10% bomb.
            # pass rule based valid actions to the exploration step
            # out of experiements, correcting actions afterward, gets bad results
            # cuz it effects the choice of the q-function
            index = None
            action = None
            if len(valid_actions) > 0:
                action = random.choice(valid_actions)

            else:
                action = random.choice(action_ideas)
            for i, act in enumerate(settings.ACTIONS):
                if act == action:
                    index = i
                    break
        else:
            self.logger.debug("Querying model for action.")
            index = self.choose_action_index(game_state, Q_eval)
        action = settings.ACTIONS[index]
        # important for the learning rate decreasing
        # other decaying methods did not work well.
        q_value = settings.ACTIONS[self.choose_action_index(game_state, Q_eval)]
        #self._epsilon_update()
        # use another strategy
        self._epsilon_update_str_2()
        Q_eval.exploratory_action_taken = action != q_value
        self.logger.debug(f" action {action}")
        return action

    def choose_action_index(self, game_state, Q_eval):
        features = self.preprocessor.state_to_features(game_state)
        features = features.unsqueeze(0).to(Q_eval.device)
        actions = Q_eval.forward(features, "online")
        index = torch.argmax(actions, dim=1)
        return index
