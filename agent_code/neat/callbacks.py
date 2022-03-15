from multiprocessing.spawn import old_main_modules
from re import A
import numpy as np
import settings as s
from . import neat

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
ACTIONS = ["WAIT", "RIGHT", "LEFT", "UP", "DOWN", "BOMB"]


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
        self.neat_population = neat.load("pickle")
    except FileNotFoundError:
        self.neat_population = neat.Population(150, 995, 2000, len(ACTIONS))
        neat.save(self.neat_population, "pickle")


def act(self, game_state):
    """
    Called each game step to determine the agent's next action.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.
    """

    # Gather information about the game state
    _, score, can_bomb, agent_position = game_state["self"]
    # need_h_flip = agent_position[0] > 9
    # need_v_flip = agent_position[1] > 9
    agent_position = [
        agent_position
    ]  # pack it into array for concatenation and convinience

    bombs_map = [xy for (xy, t) in game_state["bombs"]]
    others_map = [xy for (n, s, b, xy) in game_state["others"]]
    coins_map = game_state["coins"]
    crates_map = np.isin(game_state["field"], 1)

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
    if self.train:
        action = self.neat_population.focused_sample().feed_forward(features)
    else:
        action = self.neat_population.best_genome().feed_forward(features)
    return ACTIONS[action]


def coordinates_to_field(coordinates):
    base = np.zeros((289))
    for coord in coordinates:
        base[coord[0] * 17 + coord[1]] = 1
    return squeeze_field(base)


def squeeze_field(field: np.ndarray):
    return np.compress(field_grid_inverse, field)
