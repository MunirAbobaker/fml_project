from collections import OrderedDict
from enum import Enum
import math
import numpy as np


class InnovationGenerator:
    def __init__(self):
        self.innovation = -1

    def new_id(self):
        self.innovation += 1
        return self.innovation


innovation_generator = InnovationGenerator()


def sigmoid(z):
    z = max(-60.0, min(60.0, 5.0 * z))
    return 1.0 / (1.0 + math.exp(-z))


ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
INPUT_SIZE = 995
OUTPUT_SIZE = len(ACTIONS)
MAX_HIDDEN = 2000
INPUT_IDS = list(range(INPUT_SIZE))
OUTPUT_IDS = list(range(INPUT_SIZE + MAX_HIDDEN, INPUT_SIZE + MAX_HIDDEN + OUTPUT_SIZE))
ACTIVATION = sigmoid
EXCESS_COEFFICIENT = 1
WEIGHT_DIFF_COEFFICIENT = 2
DIFFERENCE_THRESHOLD = 1.5


class NodeGene:
    def __init__(self):
        # the id should be handled by an OrderedDict in parent Genome
        # maybe it makes sense to keep track of it anyways?
        # self.id = id
        self.state = 0
        self.incoming_connections = []


class ConnectionGene:
    def __init__(self, from_: int, to_: int, weight: float, innovation: int):
        self.from_ = from_
        self.to_ = to_
        self.weight = weight
        self.innovation = innovation
        self.enabled = True


class Genome:
    def __init__(
        self,
        nodeGenes: OrderedDict[NodeGene],
        connectionGenes: dict[ConnectionGene],
    ):
        self.vestigial = False
        self.nodeGenes = nodeGenes
        self.connectionGenes = connectionGenes
        # Bind connections to their output nodes
        for connection in self.connectionGenes.values():
            if connection.enabled:
                self.nodeGenes[connection.to_].incoming_connections.append(connection)
        self.fitness = 0

    def fresh():
        nodeGenes = OrderedDict()
        for i in INPUT_IDS:
            nodeGenes[i] = NodeGene()
        for i in OUTPUT_IDS:
            nodeGenes[i] = NodeGene()
        return Genome(nodeGenes, {})

    def size(self):
        """returns the length of nodeGenes dict"""
        return len(self.nodeGenes)

    def get_hidden_nodes_ids(self):
        return list(self.nodeGenes.keys())[INPUT_SIZE:-OUTPUT_SIZE]

    def feed_forward(self, input):
        for i in INPUT_IDS:
            self.nodeGenes[i].state = input[i]
        # first process only hidden nodes:
        # we have to do some magic because we don't know how many hidden nodes, and more importnatly - which ids they have
        for id in self.get_hidden_nodes_ids():
            state = 0
            for connection in self.nodeGenes[id].incoming_connections:
                state += self.nodeGenes[connection.from_].state * connection.weight
            self.nodeGenes[id].state = ACTIVATION(state)
        # now only do output nodes, find the argmax
        action = -math.inf
        last_action_activation = 0
        for id in OUTPUT_IDS:
            state = 0
            for connection in self.nodeGenes[id].incoming_connections:
                state += self.nodeGenes[connection.from_].state * connection.weight
            state = ACTIVATION(state)
            if state > last_action_activation:
                last_action_activation = state
                action = id
        action -= INPUT_SIZE + MAX_HIDDEN
        return ACTIONS[action]

    def mutate_weights(self):
        for connection in self.connectionGenes.values():
            # 10% chance to completely mutate weight, otherwise just shake it a little
            if np.random.random() < 0.1:
                connection.weight = np.random.random() * 2 - 1
            else:
                connection.weight += np.random.normal(scale=0.5) / 10
            if connection.weight > 1:
                connection.weight = 1
            elif connection.weight < -1:
                connection.weight = -1

    def add_connection(self):
        from_candidates = INPUT_IDS + self.get_hidden_nodes_ids()
        np.random.shuffle(from_candidates)
        to_candidates = self.get_hidden_nodes_ids() + OUTPUT_IDS
        np.random.shuffle(to_candidates)
        for from_candidate in from_candidates:
            for to_candidate in to_candidates:
                # now that we have node pair, look for existing connection:
                for c in self.connectionGenes.values():
                    if c.from_ == from_candidate and c.to_ == to_candidate:
                        break
                else:  # only executed if the inner loop did NOT break
                    innov = innovation_generator.new_id()
                    connection = ConnectionGene(
                        from_candidate, to_candidate, np.random.random() * 2 - 1, innov
                    )
                    self.connectionGenes[innov] = connection
                    return
                continue  # only executed if the inner loop DID break

    def add_node(self):
        if not self.connectionGenes:
            return
        connection = np.random.choice(list(self.connectionGenes.values()))
        connection.enabled = False
        new_node = NodeGene()
        # THIS PLACE IS SUSPICIOUS AF
        _hn = self.get_hidden_nodes_ids()
        id = 0 if (not _hn) else max(self.get_hidden_nodes_ids()) + 1
        self.nodeGenes[id] = new_node
        # first half of the old connection is split into one new connection
        innov = innovation_generator.new_id()
        self.connectionGenes[innov] = ConnectionGene(
            connection.from_, id, np.random.random() * 2 - 1, innov
        )
        new_node.incoming_connections.append(self.connectionGenes[innov])
        # second half
        innov = innovation_generator.new_id()
        self.connectionGenes[innov] = ConnectionGene(
            id, connection.to_, np.random.random() * 2 - 1, innov
        )
        self.nodeGenes[connection.to_].incoming_connections.append(
            self.connectionGenes[innov]
        )
        # we need to insert the new gene BEFORE outputs, right now it's in the end
        # calling sort on this dictionary SHOULD be O(n) since we only have one element out of place, but we should double check this
        self.nodeGenes = OrderedDict(sorted(self.nodeGenes.items(), key=lambda x: x[0]))

    def mutate(self, n_times=1):
        for i in range(n_times):
            if np.random.random() < 0.8:
                self.mutate_weights()
            if np.random.random() < 0.05:
                self.add_connection()
            if np.random.random() < 0.01:
                self.add_node()

    def weight_difference(self, other_genome):
        diff = 0
        shared_connections = 0
        for (innov1, connection1) in self.connectionGenes.items():
            for (innov2, connection2) in other_genome.connectionGenes.items():
                if innov1 == innov2:
                    diff += abs(connection1.weight - connection2.weight)
                    shared_connections += 1
        if shared_connections == 0:
            return 100

        return diff / shared_connections

    def disjoint_and_excess(self, other_genome):
        match = 0
        for innov1 in self.connectionGenes.keys():
            for innov2 in other_genome.connectionGenes.keys():
                if innov1 == innov2:
                    match += 1
        return len(self.connectionGenes) + len(other_genome.connectionGenes) - 2 * match

    def compare(self, other_genome):
        return EXCESS_COEFFICIENT * self.disjoint_and_excess(other_genome) / (
            max(1, len(other_genome.connectionGenes) + len(self.connectionGenes))
        ) + WEIGHT_DIFF_COEFFICIENT * self.weight_difference(other_genome)


def crossover(parent1: Genome, parent2: Genome) -> Genome:
    # Sort parent1 and parent2 by descending fitness, otherwise just shuffle them
    if parent1.fitness > parent2.fitness:
        pass
    elif parent1.fitness < parent2.fitness:
        _t = parent1
        parent1 = parent2
        parent2 = _t
    else:
        if np.random.random() > 0.5:
            _t = parent1
            parent1 = parent2
            parent2 = _t
    offspring_nodes = parent1.nodeGenes
    offspring_connections = {}
    for innovation in parent1.connectionGenes.keys():
        connection_parent = parent1
        enabled = True
        # if another parent has that connection, inherit it with a 0.5 chance
        if innovation in parent2.connectionGenes.keys():
            if np.random.random() > 0.5:
                connection_parent = parent2
            # if it's disabled in one of the parents, disable it with 0.75 chance
            if (
                not parent1.connectionGenes[innovation].enabled
                or not parent2.connectionGenes[innovation].enabled
            ) and np.random.random() < 0.75:
                enabled = False
        offspring_connections[innovation] = connection_parent.connectionGenes[
            innovation
        ]
        offspring_connections[innovation].enabled = enabled
    offspring = Genome(offspring_nodes, offspring_connections)
    return offspring


class Population:
    def __init__(self, size, environment=None):
        self.population_iterator = 0
        self.size = size
        self.population = []
        self.species = []
        self.env = environment
        for i in range(size):
            g = Genome.fresh()
            g.mutate(n_times=INPUT_SIZE * OUTPUT_SIZE)  # a little too much?
            self.population.append(g)
        self.speciate()

    def iterate(self):
        # once the whole population has tried playing, we can generate new offsprings
        self.population_iterator += 1
        if self.population_iterator % self.size == 0:
            self.population_iterator = 0
            self.evolve()

    def focused_sample(self) -> Genome:
        """Because NEAT has to run each genome at least once, this is the function which will point to one genome in a given round
        To select next genome for the next round, we call iterate() at the end of the round, after assigning rewards
        """
        return self.population[self.population_iterator]

    def species_total_fitness(self, specie):
        total = 0
        for genome in specie:
            total += genome.fitness
        return total

    def choose_parent(self, specie):
        # https://en.wikipedia.org/wiki/Fitness_proportionate_selection
        threshold = np.random.random() * self.species_total_fitness(specie)
        total = 0
        for genome in specie:
            total += genome.fitness
            if total > threshold:
                return genome

    def speciate(self):
        for genome in self.population:
            speciated = False
            for specie in self.species:
                sample = np.random.choice(specie)
                difference = genome.compare(sample)
                if difference < DIFFERENCE_THRESHOLD:
                    specie.append(genome)
                    speciated = True
                    break
            if not speciated:
                new_species = [genome]
                self.species.append(new_species)

    def forget_dead_species(self):
        self.species = list(filter(lambda x: x != [], self.species))

    def evolve(self):
        average_fitness = self.species_total_fitness(self.population) / self.size
        self.population = []
        to_produce = self.size
        max_total_fitness = 0  # only used for logging
        for specie in self.species:
            total_fitness = self.species_total_fitness(specie)
            _average = total_fitness / len(specie)
            if _average > max_total_fitness:
                max_total_fitness = _average
            should_produce = (
                1
                if (average_fitness == 0)
                else math.ceil(total_fitness / average_fitness) * len(specie)
            )
            to_produce -= should_produce
            # if too many individuals have been created, reduce newIndividualsCount to be within the constraints of the population size
            if to_produce < 0:
                should_produce += to_produce
                to_produce = 0
            for i in range(should_produce):
                parent1 = self.choose_parent(specie)
                parent2 = self.choose_parent(specie)
                offspring = crossover(parent1, parent2)
                offspring.mutate()
                self.population.append(offspring)

        for specie in self.species:
            for genome in specie:
                genome.vestigial = True

        self.forget_dead_species()
        self.speciate()
        for specie in self.species:
            for genome in specie:
                if genome.vestigial:
                    specie.remove(genome)
        self.forget_dead_species()
        self.env.logger.info(
            "Most fittest specie in last generation achieved {} average fitness".format(
                max_total_fitness
            )
        )
