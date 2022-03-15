import pickle
from typing import List
import events as e

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
    self.actions_used = []


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
    for event in [
        e.MOVED_RIGHT,
        e.MOVED_LEFT,
        e.MOVED_DOWN,
        e.MOVED_UP,
        e.BOMB_DROPPED,
    ]:
        if event in events and event not in self.actions_used:
            self.actions_used.append(event)


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
    self.reward_accumulator += len(self.actions_used) * 100

    self.neat_population.focused_sample().fitness = max(
        self.reward_accumulator, REWARD_ACC_INITIAL
    )
    self.neat_population.iterate()
    if self.neat_population.population_iterator == self.neat_population.size - 1:
        save(self.neat_population, "pickle")
    self.reward_accumulator = REWARD_ACC_INITIAL
    self.actions_used = []


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 100,
        e.KILLED_OPPONENT: 100,
        e.INVALID_ACTION: -3,
        e.WAITED: -1,
        e.MOVED_DOWN: 5,
        e.MOVED_UP: 5,
        e.MOVED_LEFT: 5,
        e.MOVED_RIGHT: 5,
        e.BOMB_DROPPED: 5,
        e.SURVIVED_ROUND: 100,
        e.CRATE_DESTROYED: 10,
        e.COIN_FOUND: 10,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    return reward_sum
