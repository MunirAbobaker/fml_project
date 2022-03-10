from multiprocessing.spawn import old_main_modules
from re import A
import numpy as np
import settings as s
from . import neat

import pickle

field_grid = np.array(
    [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]
).ravel()
field_grid_inverse = np.isin(field_grid, 0)


def save(population, filename):
    pickle.dump(population, open(filename, "wb"))


def load(filename):
    return pickle.load(open(filename, "rb"))


def setup(self):
    """Called once before a set of games to initialize data structures etc.

    The 'self' object passed to this method will be the same in all other
    callback methods. You can assign new properties (like bomb_history below)
    here or later on and they will be persistent even across multiple games.
    You can also use the self.logger object at any time to write to the log
    file for debugging (see https://docs.python.org/3.7/library/logging.html).
    """
    self.logger.debug("Successfully entered setup code")
    try:
        self.neat_population = load("pickle")
    except FileNotFoundError:
        self.neat_population = neat.Population(50, environment=self)
        save(self.neat_population, "pickle")


def act(self, game_state):
    """
    Called each game step to determine the agent's next action.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.
    """

    # Gather information about the game state
    _, score, can_bomb, agent_position = game_state["self"]
    need_h_flip = agent_position[0] > 8
    need_v_flip = agent_position[1] > 8
    agent_position = [
        agent_position
    ]  # pack it into array for concatenation and convinience

    bombs_map = [xy for (xy, t) in game_state["bombs"]]
    others_map = [xy for (n, s, b, xy) in game_state["others"]]
    coins_map = game_state["coins"]
    crates_map = np.isin(game_state["field"], 1)

    if need_h_flip:
        agent_position = h_flip_coordinates(agent_position)
        bombs_map = h_flip_coordinates(bombs_map)
        others_map = h_flip_coordinates(others_map)
        coins_map = h_flip_coordinates(coins_map)
        crates_map = np.fliplr(crates_map)
    if need_v_flip:
        agent_position = v_flip_coordinates(agent_position)
        bombs_map = v_flip_coordinates(bombs_map)
        others_map = v_flip_coordinates(others_map)
        coins_map = v_flip_coordinates(coins_map)
        crates_map = np.flipud(crates_map)

    crates_map = crates_map.ravel()
    self_map = coordinates_to_field(agent_position)
    bombs_map = coordinates_to_field(bombs_map)
    others_map = coordinates_to_field(others_map)
    coins_map = coordinates_to_field(coins_map)

    meta_features = [
        can_bomb,
        1,  # bias fuel
    ]

    features = np.concatenate(
        (
            crates_map,
            self_map,
            bombs_map,
            others_map,
            coins_map,
            meta_features,
        )
    )

    action = self.neat_population.focused_sample().feed_forward(features)
    if need_h_flip and action == "LEFT":
        action = "RIGHT"
    elif need_h_flip and action == "RIGHT":
        action = "LEFT"
    elif need_v_flip and action == "UP":
        action = "DOWN"
    elif need_v_flip and action == "DOWN":
        action = "UP"
    return action


def coordinates_to_field(coordinates):
    base = np.zeros((289))
    for coord in coordinates:
        base[coord[0] * 17 + coord[1]] = 1
    return squeeze_field(base)


def h_flip_coordinates(coordinates):
    return [h_flip_tuple(t) for t in coordinates]


def v_flip_coordinates(coordinates):
    return [v_flip_tuple(t) for t in coordinates]


def h_flip_tuple(coord_tuple):
    return (17 - coord_tuple[0], coord_tuple[1])


def v_flip_tuple(coord_tuple):
    return (coord_tuple[0], 17 - coord_tuple[1])


def squeeze_field(field: np.ndarray):
    return np.compress(field_grid_inverse, field)
