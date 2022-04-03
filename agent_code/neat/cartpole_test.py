from neat import *
import numpy as np
from visualizer import visualize
import gym

np.random.seed(90)


def sigmoid(z):
    z = max(-60.0, min(60.0, 4.9 * z))
    return 1.0 / (1.0 + math.exp(-z))


def cap_check(val):
    if val > 1 or val < -1:
        raise ValueError


def array_cap_check(a):
    for x in a:
        cap_check(x)


def regularize(observation):
    ret = []
    ret.append(sigmoid(observation[0]))
    ret.append(sigmoid(observation[1]))
    ret.append(sigmoid(observation[2]))
    ret.append(sigmoid(observation[3]))
    ret.append(sigmoid(observation[4]))
    ret.append(sigmoid(observation[5]))
    ret.append(observation[6])
    ret.append(observation[7])
    array_cap_check(ret)
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
            genome.fitness = reward + 1000
            break


def main():
    env = gym.make("LunarLander-v2")
    pop = Population(50, 8, 20000, 4)
    for i in range(10000 - 1):
        evaluate(pop.focused_sample(), env)
        pop.iterate()
    best = pop.best_genome()
    visualize(best)
    evaluate(best, env, render=True)
    env.close()


if __name__ == "__main__":
    main()
