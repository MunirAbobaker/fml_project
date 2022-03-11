from neat import *
import numpy as np

def main():
    pop = Population(10)
    true_positive_negative = []
    for i in range(10000):
        a = np.random.randint(2)
        b = np.random.randint(2)
        ground_truth = a ^ b
        answer = pop.focused_sample().feed_forward([a,b])
        fitness = ground_truth == answer
        true_positive_negative.append(fitness)
        pop.focused_sample().fitness = fitness
        print(sum(true_positive_negative[-10:])/10)
        pop.iterate()

if __name__=="__main__":
    main()