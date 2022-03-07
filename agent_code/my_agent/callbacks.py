import os
import pickle
import random
import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

INIT_EPSILON=0.1
MIN_EPSILON=0.1
EPSILON_DECAY_RATIO=0.9
FEATURES_DIM = 2*17*17

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS), FEATURES_DIM)
        self.model = weights / weights.sum()
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    action_index = None
    if self.train and random.random() < INIT_EPSILON:
        self.logger.debug("Exploration: Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        action_index = np.random.randint(len(ACTIONS))
    else:
        self.logger.debug("Greedy: Exploitation")
        features = state_to_features(game_state)
        self.logger.debug(f"feature dim: {features.shape}")
        Q = np.dot(self.model,features)
        action_index = np.argmax(Q)
    return ACTIONS[np.argmax(action_index)]


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None
    # For example, you could construct several channels of equal shape, ...
    channels = []
    arena_channel = game_state["field"]
    coins = game_state['coins'] # throws error, has another shape
    explosion_channel = game_state['explosion_map']
    _, score, bombs_left, (x, y) = game_state['self']
    bombs = game_state["bombs"]
    others_channel = np.zeros_like(arena_channel)
    bombs_map = np.zeros_like(arena_channel)
    self_channel = np.zeros_like(arena_channel)
    coins_channel = np.zeros_like(arena_channel)

    for (xb, yb), t in bombs:
        bombs_map[xb, yb] = t

    self_channel[x, y] = 1
    channels.append(arena_channel)
    channels.append(explosion_channel)
    #channels.extend([arena_channel, explosion_channel, bombs_map, self_channel])
    # concatenate them as a feature tensor (they must have the same shape),
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)


