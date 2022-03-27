import numpy as np
import torch

from . import settings


class Preprocessor(object):
    def __init__(self, logger):
        self.logger = logger

    def state_to_features(self, game_state: dict) -> torch.Tensor:
        """
        *This is not a required function, but an idea to structure your code.*
        Converts the game state to the input of your model, i.e.
        a feature vector.
        You can find out about the state of the game environment via game_state,
        which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
        what it contains.
        :param game_state:  A dictionary describing the current game board.
        :return: torch.Tensor
        """

        # This is the dict before the game begins and after it ends
        if game_state is None:
            return None
        # For example, you could construct several channels of equal shape, ...
        channels = []

        arena_channel = game_state["field"]
        bombs = game_state["bombs"]
        coins = game_state['coins']
        explosion_channel = game_state['explosion_map']
        _, score, bombs_left, (x, y) = game_state['self']

        others_channel = np.zeros_like(arena_channel)
        bombs_channel = np.zeros_like(arena_channel)
        self_channel = np.zeros_like(arena_channel)
        coins_channel = np.zeros_like(arena_channel)

        for x_c, y_c in coins:
            coins_channel[x_c, y_c] = 1

        for (xb, yb), t in bombs:
            bombs_channel[xb, yb] = t

        self_channel[game_state["self"][3]] = 1

        channels.append(explosion_channel)
        channels.append(self_channel)
        channels.append(coins_channel)
        # channels.append(others_channel)
        # concatenate them as a feature tensor (they must have the same shape),
        stacked_channels = np.stack(channels)
        # and return them as a vector, stacked_channels.reshape(-1)
        if settings.NUMBER_OF_CHANNELS != stacked_channels.shape[0]:
            print("dev: remove/ add one element to the array")
        features = torch.from_numpy(stacked_channels).float()
        return features