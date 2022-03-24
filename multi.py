import os
from argparse import ArgumentParser
from pathlib import Path
import threading
from time import sleep, time

import settings as s
from environment import BombeRLeWorld
from fallbacks import pygame, LOADED_PYGAME

ESCAPE_KEYS = (pygame.K_q, pygame.K_ESCAPE)


class WorldThread(threading.Thread):
    def __init__(self, threadID, world, n_rounds):
        thread = threading.Thread.__init__(self)
        self.threadID = threadID
        self.world = world
        self.n_rounds = n_rounds

    def run(self):
        for i in range(self.n_rounds):
            self.world.new_round()

    def join(self):
        self.world.end()
        super().join()


def world_controller(worlds, n_rounds, /, threads):
    threads = []
    i = 0
    for w in worlds:
        t = WorldThread(i, w, n_rounds / threads)
        threads.append(t)
        i += 1
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def main(argv=None):
    parser = ArgumentParser()

    subparsers = parser.add_subparsers(dest="command_name", required=True)

    # Run arguments
    play_parser = subparsers.add_parser("play")
    agent_group = play_parser.add_mutually_exclusive_group()
    agent_group.add_argument(
        "--my-agent",
        type=str,
        help="Play agent of name ... against three rule_based_agents",
    )
    agent_group.add_argument(
        "--agents",
        type=str,
        nargs="+",
        default=["rule_based_agent"] * s.MAX_AGENTS,
        help="Explicitly set the agent names in the game",
    )
    play_parser.add_argument(
        "--train",
        default=0,
        type=int,
        choices=[0, 1, 2, 3, 4],
        help="First … agents should be set to training mode",
    )
    play_parser.add_argument(
        "--continue-without-training", default=False, action="store_true"
    )
    # play_parser.add_argument("--single-process", default=False, action="store_true")

    play_parser.add_argument("--scenario", default="classic", choices=s.SCENARIOS)

    play_parser.add_argument(
        "--seed",
        type=int,
        help="Reset the world's random number generator to a known number for reproducibility",
    )

    play_parser.add_argument(
        "--n-rounds", type=int, default=10, help="How many rounds to play"
    )
    play_parser.add_argument(
        "--save-replay",
        const=True,
        default=False,
        action="store",
        nargs="?",
        help="Store the game as .pt for a replay",
    )
    play_parser.add_argument("--match-name", help="Give the match a name")

    play_parser.add_argument(
        "--silence-errors",
        default=False,
        action="store_true",
        help="Ignore errors from agents",
    )

    group = play_parser.add_mutually_exclusive_group()
    group.add_argument(
        "--skip-frames",
        default=False,
        action="store_true",
        help="Play several steps per GUI render.",
    )
    group.add_argument(
        "--no-gui",
        default=False,
        action="store_true",
        help="Deactivate the user interface and play as fast as possible.",
    )

    # Replay arguments
    replay_parser = subparsers.add_parser("replay")
    replay_parser.add_argument("replay", help="File to load replay from")

    # Interaction
    for sub in [play_parser, replay_parser]:
        sub.add_argument(
            "--turn-based",
            default=False,
            action="store_true",
            help="Wait for key press until next movement",
        )
        sub.add_argument(
            "--update-interval",
            type=float,
            default=0.1,
            help="How often agents take steps (ignored without GUI)",
        )
        sub.add_argument(
            "--log-dir", default=os.path.dirname(os.path.abspath(__file__)) + "/logs"
        )
        sub.add_argument(
            "--save-stats",
            const=True,
            default=False,
            action="store",
            nargs="?",
            help="Store the game results as .json for evaluation",
        )

        # Video?
        sub.add_argument(
            "--make-video",
            const=True,
            default=False,
            action="store",
            nargs="?",
            help="Make a video from the game",
        )

    args = parser.parse_args(argv)
    if args.command_name == "replay":
        args.no_gui = False
        args.n_rounds = 1
        args.match_name = Path(args.replay).name

    has_gui = not args.no_gui

    THREADS = 3

    # Initialize environment and agents
    if args.command_name == "play":

        worlds = []
        for _ in range(THREADS):
            agents = []
            agents.append((args.my_agent, True))
            rule_agents = ["rule_based_agent"] * (s.MAX_AGENTS - 1)
            for agent_name in rule_agents:
                agents.append((agent_name, False))
            w = BombeRLeWorld(args, agents, 150)
            worlds.append(w)
        every_step = not args.skip_frames
    else:
        raise ValueError(f"Unknown command {args.command_name}")

    world_controller(
        worlds,
        args.n_rounds,
        threads=THREADS,
    )


if __name__ == "__main__":
    main()
