import math
from multiprocessing.spawn import old_main_modules
from re import A
import numpy as np
import settings as s
from . import neat

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
        self.neat_population.logger = self.logger
    except FileNotFoundError:
        self.neat_population = neat.Population(
            250, 35, 1000000, len(ACTIONS), logger=self.logger
        )
        neat.save(self.neat_population, "pickle")


def manhattan(origin, destination):
    return abs(destination[0] - origin[0]) + abs(destination[1] - origin[1])


def euclidean(origin, destination):
    return math.sqrt(
        (origin[0] - destination[0]) ** 2 + (origin[1] - destination[1]) ** 2
    )


def compute_valid_moves(agent_position, field):
    x, y = agent_position[0], agent_position[1]
    ret = []
    ret.append((field[x + 1][y] == 0) * 2 - 1)
    ret.append((field[x - 1][y] == 0) * 2 - 1)
    ret.append((field[x][y + 1] == 0) * 2 - 1)
    ret.append((field[x][y - 1] == 0) * 2 - 1)
    return ret


def compute_bomb_appeal(agent_position, field):
    x, y = agent_position[0], agent_position[1]
    crates_would_get_bombed = 0
    for i in range(1, 4):
        f = field[x + i][y]
        if f == -1:
            break
        else:
            crates_would_get_bombed += f
    for i in range(1, 4):
        f = field[x - i][y]
        if f == -1:
            break
        else:
            crates_would_get_bombed += f
    for i in range(1, 4):
        f = field[x][y + i]
        if f == -1:
            break
        else:
            crates_would_get_bombed += f
    for i in range(1, 4):
        f = field[x][y - i]
        if f == -1:
            break
        else:
            crates_would_get_bombed += f
    crates_would_get_bombed /= (
        12  # that's max number of crates that theoretically could be bombed
    )
    return crates_would_get_bombed


def act(self, game_state):
    """
    Called each game step to determine the agent's next action.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.
    """

    _, score, can_bomb, agent_position = game_state["self"]

    others_map = [xy for (n, s, b, xy) in game_state["others"]]
    bombs_map = [(xy, t) for (xy, t) in game_state["bombs"]]

    enemy_features = []  # len 9
    blank = 3 - len(others_map)
    holder = []
    for enemy_position in others_map:
        t = []
        euc = euclidean(agent_position, enemy_position)
        sin = (enemy_position[1] - agent_position[1]) / euc if euc != 0 else 0
        cos = (enemy_position[0] - agent_position[0]) / euc if euc != 0 else 0
        t.append(sin)
        t.append(cos)
        t.append(1 - manhattan(agent_position, enemy_position) / 28)
        holder.append(t)
    holder.sort(key=lambda t: t[2])
    for t in holder:
        enemy_features += t
    for i in range(blank):
        enemy_features += [0, 0, 0]

    bomb_features = []  # len 16
    blank = 4 - len(bombs_map)
    holder = []
    self.bomb_nearby = False  # used in training
    for bomb_tuple in bombs_map:
        tup = []
        bomb_position = bomb_tuple[0]
        t = 1 - bomb_tuple[1] / 4
        euc = euclidean(agent_position, bomb_position)
        sin = (bomb_position[1] - agent_position[1]) / euc if euc != 0 else 0
        cos = (bomb_position[0] - agent_position[0]) / euc if euc != 0 else 0
        tup.append(sin)
        tup.append(cos)
        tup.append(1 - manhattan(agent_position, bomb_position) / 28)
        tup.append(t)
        holder.append(tup)
    holder.sort(key=lambda t: t[2])
    for tup in holder:
        bomb_features += tup
    for i in range(blank):
        bomb_features += [0, 0, 0, 0]

    # len 3
    nearest_distance = 28
    nearest_coin = None
    for coin_position in game_state["coins"]:
        dist = manhattan(agent_position, coin_position)
        if dist <= nearest_distance:
            nearest_coin = coin_position
            nearest_distance = dist
    if not nearest_coin:
        coin_features = [0, 0, 0]
    else:
        euc = euclidean(agent_position, nearest_coin)
        sin = (nearest_coin[1] - agent_position[1]) / euc if euc != 0 else 0
        cos = (nearest_coin[0] - agent_position[0]) / euc if euc != 0 else 0
        coin_features = [sin, cos, 1 - nearest_distance / 28]

    box_and_wall = compute_valid_moves(agent_position, game_state["field"])
    explosions_blocked = compute_valid_moves(
        agent_position, game_state["explosion_map"]
    )
    valid_moves = []
    for i in range(4):
        valid_moves.append(box_and_wall[i] + explosions_blocked[i])
        valid_moves[i] = 1 if valid_moves[i] == 2 else -1

    # this computes how many boxes will be exploded if we place a bomb now, normalized
    bomb_appeal = compute_bomb_appeal(agent_position, game_state["field"])

    features = np.concatenate(
        (
            enemy_features,
            bomb_features,
            coin_features,
            valid_moves,
            [bomb_appeal, can_bomb, 1],
        )
    )
    if self.train:
        action = self.neat_population.focused_sample().feed_forward(features)
    else:
        action = self.neat_population.best_genome().feed_forward(features)
    return ACTIONS[action]
