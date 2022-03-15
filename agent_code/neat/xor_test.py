from neat import *
import numpy as np

np.random.seed(90)


def generate_data():
    a = np.random.randint(2)
    b = np.random.randint(2)
    ground_truth = a ^ b
    return a, b, ground_truth


def main():
    pop_size = 25
    iterations = 500
    pop = Population(pop_size, 2, 2, 2)
    true_positive_negative = []
    a, b, ground_truth = generate_data()
    for i in range(iterations):
        if i % pop_size == 0:
            a, b, ground_truth = generate_data()
        answer = pop.focused_sample().feed_forward([a, b])
        fitness = ground_truth == answer
        true_positive_negative.append(fitness)
        pop.focused_sample().fitness = fitness
        print(sum(true_positive_negative[-10:]) / 10)
        pop.iterate()
    ins = [[0, 0], [1, 1], [0, 1], [1, 0]]
    outs = [0, 0, 1, 1]
    best = pop.best_genome()
    for i in range(4):
        print(best.feed_forward(ins[i]) == outs[i])


if __name__ == "__main__":
    main()
