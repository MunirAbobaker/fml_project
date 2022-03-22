import math
import os
import random
import sys
from collections import deque

import numpy as np
import torch
from tensorflow import keras

from . import settings
from .model import  DeepQNetwork


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
    self.coordinate_history = deque([], 20)
    if self.train or not os.path.isfile(settings.LOAD_PATH):
        self.logger.info("Setting up model from scratch.")
        self.Q_eval = DeepQNetwork(0.99, input_dims=settings.INPUT_DIMENSIONS,
                                   n_actions=len(settings.ACTIONS),
                                   fc1_dims=256, fc2_dims=256, that=self).float()
    else:
        self.logger.info("Loading model from saved state.")
        #with open("my-saved-model.pt", "rb") as file:
        #    self.model = pickle.load(file)

    self.burnin = 1e4  # min. experiences before training
    self.learn_every = 3  # no. of experiences between updates to Q_online
    self.sync_every = 1e4  # no. of experiences between Q_target & Q_online sync
    self.save_every = 5e5  # no. of experiences between saving Q_eval
    self.exploration_rate = 0.01
    self.exploration_rate_decay = 0.99999975
    self.exploration_rate_min = 0.1
    self.ignore_others_timer = 0

    ####################################
    ## Greedy decay
    ####################################
    self.epsilon = 1.0
    self.init_epsilon = 1.0
    self.decay_steps = 20000
    self.min_epsilon = 0.1
    self.epsilons = 0.01 / np.logspace(-2, 0, self.decay_steps, endpoint=False) - 0.01
    self.epsilons = self.epsilons * (self.init_epsilon - self.min_epsilon) + self.min_epsilon
    self.t = 0
    self.exploratory_action_taken = None

def epsilon_update(self):
    self.epsilon = self.min_epsilon if self.t >= self.decay_steps else self.epsilons[self.t]
    self.t += 1
    return self.epsilon

def choose_rule_based_valid_action(self, game_state):
    # Gather information about the game state
    arena = game_state['field']
    _, score, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state['others']]
    coins = game_state['coins']
    bomb_map = np.ones(arena.shape) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)

    # If agent has been in the same location three times recently, it's a loop
    # if self.coordinate_history.count((x, y)) > 2:
    #    self.ignore_others_timer = 5
    # else:
    #    self.ignore_others_timer -= 1
    self.coordinate_history.append((x, y))

    # Check which moves make sense at all
    directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    # [0: 'UP', 1:'RIGHT', 2:'DOWN', 3:'LEFT', 4:'WAIT', 5:'BOMB']
    valid_tiles, valid_actions, valid_actions_one_hot = [], [], [0 for _ in range(len(settings.ACTIONS))]
    for d in directions:
        if ((arena[d] == 0) and
                (game_state['explosion_map'][d] < 1) and
                (bomb_map[d] > 0) and
                (not d in others) and
                (not d in bomb_xys)):
            valid_tiles.append(d)
    if (x - 1, y) in valid_tiles:
        valid_actions.append('LEFT')
        valid_actions_one_hot[3] = 1
    if (x + 1, y) in valid_tiles:
        valid_actions.append('RIGHT')
        valid_actions_one_hot[1] = 1
    if (x, y - 1) in valid_tiles:
        valid_actions.append('UP')
        valid_actions_one_hot[0] = 1
    if (x, y + 1) in valid_tiles:
        valid_actions.append('DOWN')
        valid_actions_one_hot[2] = 1
    if (x, y) in valid_tiles:
        valid_actions.append('WAIT')
        valid_actions_one_hot[4] = 1
    # Disallow the BOMB action if agent dropped a bomb in the same spot recently
    # if (bombs_left > 0) and (x, y) not in self.bomb_history: valid_actions.append('BOMB')
    self.logger.debug(f'Valid actions: {valid_actions}')


    return valid_actions

def look_for_targets(free_space, start, targets, logger=None):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards closest target or towards tile closest to any target.
    """
    if len(targets) == 0: return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        random.shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger: logger.debug(f'Suitable target found at {best}')
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start: return current
        current = parent_dict[current]

def act(self, game_state: dict) -> str:
    self.exploratory_action_taken = False

    valid_actions = choose_rule_based_valid_action(self, game_state)
    self.logger.debug(f" valid_actions {valid_actions}")
    ########################
    arena = game_state['field']
    _, score, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    coins = game_state['coins']
    others = [xy for (n, s, b, xy) in game_state['others']]
    # Collect basic action proposals in a queue
    # Later on, the last added action that is also valid will be chosen
    action_ideas = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    random.shuffle(action_ideas)

    # Compile a list of 'targets' the agent should head towards
    cols = range(1, arena.shape[0] - 1)
    rows = range(1, arena.shape[0] - 1)
    dead_ends = [(x, y) for x in cols for y in rows if (arena[x, y] == 0)
                 and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(0) == 1)]
    crates = [(x, y) for x in cols for y in rows if (arena[x, y] == 1)]
    targets = coins + dead_ends + crates
    # Add other agents as targets if in hunting mode or no crates/coins left
    #if self.ignore_others_timer <= 0 or (len(crates) + len(coins) == 0):
       # targets.extend(others)
    targets = [targets[i] for i in range(len(targets)) if targets[i] not in bomb_xys]

    # Take a step towards the most immediately interesting target
    free_space = arena == 0
    if self.ignore_others_timer > 0:
        for o in others:
            free_space[o] = False
    d = look_for_targets(free_space, (x, y), targets, self.logger)
    if d == (x, y - 1): action_ideas.append('UP')
    if d == (x, y + 1): action_ideas.append('DOWN')
    if d == (x - 1, y): action_ideas.append('LEFT')
    if d == (x + 1, y): action_ideas.append('RIGHT')
    # Add proposal to drop a bomb if at dead end
    if (x, y) in dead_ends:
        action_ideas.append('BOMB')
    # Add proposal to drop a bomb if touching an opponent
    if len(others) > 0:
        if (min(abs(xy[0] - x) + abs(xy[1] - y) for xy in others)) <= 1:
            action_ideas.append('BOMB')
    # Add proposal to drop a bomb if arrived at target and touching crate
    if d == (x, y) and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(1) > 0):
        action_ideas.append('BOMB')

    # Add proposal to drop a bomb if at dead end
    if (x, y) in dead_ends:
        action_ideas.append('BOMB')

        # Add proposal to run away from any nearby bomb about to blow
    for (xb, yb), t in bombs:
        if (xb == x) and (abs(yb - y) < 4):
            # Run away
            if (yb > y): action_ideas.append('UP')
            if (yb < y): action_ideas.append('DOWN')
            # If possible, turn a corner
            action_ideas.append('LEFT')
            action_ideas.append('RIGHT')
        if (yb == y) and (abs(xb - x) < 4):
            # Run away
            if (xb > x): action_ideas.append('LEFT')
            if (xb < x): action_ideas.append('RIGHT')
            # If possible, turn a corner
            action_ideas.append('UP')
            action_ideas.append('DOWN')
    # Try random direction if directly on top of a bomb
    for (xb, yb), t in bombs:
        if xb == x and yb == y:
            action_ideas.extend(action_ideas[:4])
    self.logger.debug(f"action ideas after bombs {action_ideas}")

    if self.train and random.random() < self.epsilon:
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

        #actions_range = [i for i in range(len(settings.ACTIONS))]
        #index = np.random.choice(actions_range, p=[.3, .2, .2, .1, .1, .1])
    else:
        self.logger.debug("Querying model for action.")
        index = choose_action_index(self, game_state)
    action = settings.ACTIONS[index]
    epsilon_update(self)
    # important for the learning rate decreasing
    # other decaying methods did not work well.
    q_value = settings.ACTIONS[choose_action_index(self, game_state)]
    self.exploratory_action_taken = action != q_value
    self.logger.debug(f" action {action}")


    #############

    """if len(valid_actions) > 2 and "WAIT" in valid_actions:
        self.logger.debug("remove wait")
        #valid_actions.remove("WAIT")
    if action in valid_actions and action == "BOMB":
        while len(valid_actions) > 0:
            if len(valid_actions) > 2 and "WAIT" in valid_actions:
                self.logger.debug("remove wait")
                valid_actions.remove("WAIT")
            if len(valid_actions) > 2 and "BOMB" in valid_actions:
                self.logger.debug("remove BOMB")
                valid_actions.remove("BOMB")

            a = valid_actions.pop()
            self.logger.debug(f"while loop action {a}")
            if a in valid_actions:
                # Keep track of chosen action for cycle detection
                #if a == 'BOMB':
                    #self.bomb_history.append((x, y))

                return a """


    return action



def choose_action_index(self, game_state):
    features = state_to_features(game_state)
    features = features.unsqueeze(0).to(self.Q_eval.device)
    actions = self.Q_eval.forward(features, "online")
    index = torch.argmax(actions, dim=1)
    return index


def state_to_features(game_state: dict) -> torch.Tensor:
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
    #channels.append(others_channel)
    # concatenate them as a feature tensor (they must have the same shape),
    stacked_channels = np.stack(channels)
    # and return them as a vector, stacked_channels.reshape(-1)
    if settings.NUMBER_OF_CHANNELS != stacked_channels.shape[0]:
        print("dev: remove/ add one element to the array")
    features = torch.from_numpy(stacked_channels).float()
    return features