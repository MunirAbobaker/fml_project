import math
import pickle
from typing import List
import events as e
from .callbacks import manhattan

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
REWARD_ACC_INITIAL = 1


def save(population, filename):
    pickle.dump(population, open(filename, "wb"))


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # initialized to 1, otherwise we get zero divison errors when computing stuff with fitnesses of genomes
    self.reward_accumulator = REWARD_ACC_INITIAL
    self.actions_used = {
        e.MOVED_RIGHT: 0,
        e.MOVED_LEFT: 0,
        e.MOVED_DOWN: 0,
        e.MOVED_UP: 0,
        e.BOMB_DROPPED: 0,
    }
    self.step_counter = 0
    self.last_recorded_position = None


def game_events_occurred(
    self,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: List[str],
):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.reward_accumulator += reward_from_events(self, events)
    if self.step_counter % 5 == 0:
        self.reward_accumulator += delta_position(self, new_game_state)

    self.step_counter += 1
    for event in events:
        if event in self.actions_used.keys():
            self.actions_used[event] += 1


def delta_position(self, game_state):
    ret = 0
    current_pos = game_state["self"][3]
    if self.last_recorded_position != None:
        ret += manhattan(current_pos, self.last_recorded_position) * 20
    self.last_recorded_position = current_pos
    return ret


def measure_diversity_of_actions(actions):
    s = 0
    for action_count in actions:
        s += math.sqrt(action_count)
    return s ** 2


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(
        f'Encountered event(s) {", ".join(map(repr, events))} in final step'
    )

    # Reward a network for trying different actions
    self.reward_accumulator += (
        measure_diversity_of_actions(self.actions_used.values()) * 20
    )

    self.neat_population.focused_sample().fitness = max(
        self.reward_accumulator, REWARD_ACC_INITIAL
    )
    self.neat_population.iterate()
    if self.neat_population.population_iterator == self.neat_population.size - 1:
        save(self.neat_population, "pickle")
    self.reward_accumulator = REWARD_ACC_INITIAL
    self.step_counter = 0
    self.last_recorded_position = None
    self.actions_used = {
        e.MOVED_RIGHT: 0,
        e.MOVED_LEFT: 0,
        e.MOVED_DOWN: 0,
        e.MOVED_UP: 0,
        e.BOMB_DROPPED: 0,
    }


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 7500,
        e.KILLED_OPPONENT: 5000,
        e.KILLED_SELF: -1000,
        e.GOT_KILLED: -5000,
        e.INVALID_ACTION: -1,
        e.WAITED: 0,
        e.MOVED_DOWN: 1,
        e.MOVED_UP: 1,
        e.MOVED_LEFT: 1,
        e.MOVED_RIGHT: 1,
        e.BOMB_DROPPED: 5000,
        e.SURVIVED_ROUND: 15000,
        e.CRATE_DESTROYED: 1000,
        e.COIN_FOUND: 1000,
    }
    reward_sum = 1
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    return reward_sum
