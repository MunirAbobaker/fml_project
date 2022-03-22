import pickle
import random
from collections import namedtuple, deque
from typing import List

import events as e
from .callbacks import state_to_features
from . import settings
import torch

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))



def setup_training(self):
    """
    Initialise self for training purpose.
    This is called after `setup` in callbacks.py.
    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=settings.MEMORY_SIZE)
    self.optimizer = torch.optim.Adam(self.Q_eval.parameters(), lr=0.00025)
    self.loss_fn = torch.nn.SmoothL1Loss()

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
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
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Idea: Add your own events to hand out rewards
    # if ...:
    #     events.append(PLACEHOLDER_EVENT)

    # state_to_features is defined in callbacks.py
    if old_game_state is None or new_game_state is None:
        return
    self.curr_step = old_game_state["step"]
    round = old_game_state["round"]

    if self.curr_step % self.sync_every == 0:
        self.sync_Q_target()

    if self.curr_step % self.save_every == 0:
        self.save()

    if self.curr_step < self.burnin:
        return

    if self.curr_step % self.learn_every != 0:
        return

    self.transitions.append(
        Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state),
                   reward_from_events(self, events)))
    self.logger.debug(f"transitions size {self.transitions}")

    state, next_state, action, reward = racall(self)
    td_est = td_estimate(self, state, action)
    td_targ = td_target(self, reward, next_state)
    loss = update_online(self, td_est, td_targ)
    self.loss[round].append(loss)


def td_estimate(self, state, action):
    current_Q = self.Q_eval(state, "online")[
        torch.arange(0, settings.BATCH_SIZE, dtype=torch.long),
        action]
    return current_Q


def update_online(self, td_estimate, td_target):
    loss = self.loss_fn(td_estimate, td_target)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    return loss.item()


def update_target(self):
    self.Q_eval.target.load_state_dict(self.Q_eval.online.state_dict())


@torch.no_grad()
def td_target(self, reward, next_state):
    next_state_Q = self.Q_eval(next_state, "online")
    best_action = torch.argmax(next_state_Q, axis=1)
    next_Q = self.model(next_state, model="target")[
        torch.arange(0, settings.BATCH_SIZE, dtype=torch.long),
        best_action]
    return (reward + settings.GAMMA * next_Q).float()


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.
    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.
    :param self: The same object that is passed to all of your callbacks.
    """
    update_target(self)
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(
        Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pass
        pickle.dump(self.Q_eval, file)


def racall(self):
    print(f"size of transitions {self.transitions}")
    batch = random.sample(self.transitions, settings.BATCH_SIZE)

    state, next_state, action, reward, done = map(torch.stack, zip(*batch))
    return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*
    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    self.logger.debug(f"event is {events}")

    game_rewards = {
        e.COIN_COLLECTED: 10,
        e.SURVIVED_ROUND: 1,
        #e.KILLED_OPPONENT: 1,
        #e.KILLED_SELF: -1,
        e.CRATE_DESTROYED: 1,
        #e.WAITED: -2,
        #e.INVALID_ACTION: -50,
        #e.BOMB_DROPPED: 0.2
        #PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum