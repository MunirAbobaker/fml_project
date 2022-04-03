from neat import *
import numpy as np
from visualizer import visualize
import gym

np.random.seed(90)


def regularize(observation):
    ret = [observation[0] + 0.3, observation[1] * 14]
    return ret


def evaluate(genome: Genome, env: gym.Env, render=False):
    observation = env.reset()
    observation = regularize(observation)
    for t in range(1000):
        action = genome.feed_forward(observation)
        if render:
            env.render("human")
            print(action)
        observation, reward, done, info = env.step(action)
        observation = regularize(observation)
        if done:
            print("Episode finished after {} timesteps".format(t))
            genome.fitness = reward
            break


def main():
    env = gym.make("Ant-v2")
    pop = Population(25, 2, 20000, 3)
    for i in range(500 - 1):
        evaluate(pop.focused_sample(), env)
        pop.iterate()
    best = pop.best_genome()
    visualize(best)
    evaluate(best, env, render=True)
    env.close()


if __name__ == "__main__":
    main()
