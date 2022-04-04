from neat import *
import numpy as np
from visualizer import visualize
import gym

np.random.seed(90)


def sigmoid(z):
    z = max(-60.0, min(60.0, 4.9 * z))
    return 1.0 / (1.0 + math.exp(-z))


mins = [math.inf]*8
maxs = [-math.inf]*8

def regularize(observation):
    ret = []
    for i in range(8):
        if observation[i] > maxs[i]:
            maxs[i] = observation[i]
        if observation[i] < mins[i]:
            mins[i] = observation[i]
        ret.append(observation[i]/(maxs[i]-mins[i]))
    ret.append(1)
    return ret


def evaluate(genome: Genome, env: gym.Env, render=False):
    observation = env.reset()
    observation = regularize(observation)
    for t in range(1000):
        action = genome.feed_forward(observation)
        if render:
            env.render("human")
        observation, reward, done, info = env.step(action)
        observation = regularize(observation)
        if done:
            print("Episode finished after {} timesteps".format(t))
            genome.fitness = reward + 100
            break


def main():
    env = gym.make("LunarLander-v2")
    pop = Population(50, 9, 20000, 4)
    for i in range(1000 - 1):
        evaluate(pop.focused_sample(), env)
        pop.iterate()
    best = pop.best_genome()
    visualize(best)
    evaluate(best, env, render=True)
    env.close()


if __name__ == "__main__":
    main()
