from neat import *
import numpy as np
from visualizer import visualize

np.random.seed(90)


def generate_data():
    a = np.random.randint(2)
    b = np.random.randint(2)
    ground_truth = a ^ b
    return a, b, ground_truth


def generate_batch(size):
    batch = []
    for i in range(size):
        batch.append(generate_data())
    return batch


def main():
    pop_size = 50
    iterations = 10000 - 1
    batch_size = 50
    pop = Population(pop_size, 3, 2000, 2)
    batch = []
    for i in range(iterations):
        print(i)
        if i % pop_size == 0:
            batch = generate_batch(batch_size)
        fitness = 0
        for each in batch:
            features = [each[0], each[1], 1]  # added 1 to fuel bias
            answer = pop.focused_sample().feed_forward(features)
            fitness += answer == each[-1]
        if fitness == batch_size:
            break
        pop.focused_sample().fitness = fitness
        pop.iterate()

    ins = [[0, 0, 1], [1, 1, 1], [0, 1, 1], [1, 0, 1]]
    outs = [0, 0, 1, 1]
    best = pop.best_genome()
    visualize(best)
    for i in range(4):
        print(best.feed_forward(ins[i]) == outs[i])


if __name__ == "__main__":
    main()
