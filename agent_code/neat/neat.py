from collections import OrderedDict
import math
import pickle
import numpy as np
import pygame
from numpy import clip
from . import visualizer


def save(population, filename):
    pickle.dump(population, open(filename, "wb"))


def load(filename):
    return pickle.load(open(filename, "rb"))


def sigmoid(z):
    z = max(-60.0, min(60.0, 4.9 * z))
    return 1.0 / (1.0 + math.exp(-z))


def szudzik_pair(a, b):
    return (a * a) + a + b if a >= b else (b * b) + a


# python can not pickle lambda functions, so we gotta define this one for filter >:|
def non_vestigial(g):
    return not g.vestigial


# Meta params:
ACTIVATION = sigmoid
EXCESS_COEFFICIENT = 1  # How important is difference in topology for speciation
WEIGHT_DIFF_COEFFICIENT = 0.4  # How important is difference in weights for speciation
DIFFERENCE_THRESHOLD = (
    1.5  # The threshold which needs to be exceeded to be put into a new specie
)
COMPLETELY_MUTATE_WEIGHT_CHANCE = 0.1  # Chance to randomize the weight instead of slightly modifying it when mutate_weights is triggered
MUTATE_WEIGHTS_CHANCE = 0.8
ADD_CONNECTION_CHANCE = 0.10
ADD_NODE_CHANCE = 0.06
INTERSPECIES_MATING_CHANCE = 0.010
MUTATION_BASE_AMPLIFIER = 1 / ADD_CONNECTION_CHANCE


class NodeGene:
    def __init__(self):
        self.state = 0
        self.incoming_connections = []

    def deepcopy_dict(arr: OrderedDict):
        copy = OrderedDict()
        for (key, node) in arr.items():
            copy[key] = node.copy()
        return copy

    def copy(self):
        c = NodeGene()
        c.incoming_connections = self.incoming_connections.copy()
        return c


class ConnectionGene:
    def __init__(self, from_: int, to_: int, weight: float):
        self.from_ = from_
        self.to_ = to_
        self.weight = weight
        self.unique_id = szudzik_pair(from_, to_)
        self.enabled = True

    def copy(self):
        return ConnectionGene(self.from_, self.to_, self.weight)


class Genome:
    def __init__(
        self,
        nodeGenes: OrderedDict[NodeGene],
        connectionGenes: dict[ConnectionGene],
        master_population,
    ):
        self.nodeGenes = nodeGenes
        self.connectionGenes = connectionGenes
        self.master_population = master_population

        self.vestigial = False
        self.fitness = 0
        # Bind connections to their output nodes TODO check if we need this actually
        for connection in self.connectionGenes.values():
            if connection.enabled:
                self.nodeGenes[connection.to_].incoming_connections.append(connection)

    def visualize(self, surface: pygame.Surface):
        node_size = 10
        padding = 20
        hidden_count = (
            len(self.nodeGenes)
            - self.master_population.INPUT_SIZE
            - self.master_population.OUTPUT_SIZE
        )

        input_h = math.ceil(math.sqrt(self.master_population.INPUT_SIZE * 2))
        input_w = input_h // 2
        hidden_h = math.ceil(math.sqrt((hidden_count * 2)))
        hidden_w = hidden_h // 2
        output_h = self.master_population.OUTPUT_SIZE
        output_w = 1

        positions = {}
        node_center = node_size // 2

        for x in range(input_w):
            for y in range(input_h):
                positions[x * input_h + y] = (
                    x * (node_size + padding) + padding,
                    y * (node_size + padding) + padding,
                )
        hidden_offset = input_w * (node_size + padding) + padding * 5

        hidden_iterator_w = 0
        hidden_iterator_h = 0

        for id in self.get_hidden_nodes_ids():
            positions[id] = (
                hidden_iterator_w * (node_size + padding) + hidden_offset,
                hidden_iterator_h * (node_size + padding) + padding * 1.5,
            )
            hidden_iterator_h += 1
            if hidden_iterator_h > hidden_h:
                hidden_iterator_h = 0
                hidden_iterator_w += 1

        output_offset = hidden_offset + hidden_w * (node_size + padding) + padding * 5
        for x in range(output_w):
            for y in range(output_h):
                positions[
                    x * output_h
                    + y
                    + self.master_population.INPUT_SIZE
                    + self.master_population.MAX_HIDDEN
                ] = (
                    x * (node_size + padding) + output_offset + padding,
                    y * (node_size + padding) + padding,
                )
        for id in self.nodeGenes.keys():
            pygame.draw.circle(surface, (200, 0, 200), positions[id], node_size)
        for connection in self.connectionGenes.values():
            color = (
                (20, 20, 20)
                if not connection.enabled
                else (
                    pygame.Color(255, 0, 0).lerp(
                        (0, 255, 0), (connection.weight + 1) / 2
                    )
                )
            )
            pygame.draw.aaline(
                surface,
                color,
                positions[connection.from_],
                positions[connection.to_],
            )

    def fresh(master_population):
        nodeGenes = OrderedDict()
        for i in master_population.INPUT_IDS:
            nodeGenes[i] = NodeGene()
        for i in master_population.OUTPUT_IDS:
            nodeGenes[i] = NodeGene()
        return Genome(nodeGenes, {}, master_population)

    def size(self):
        return len(self.nodeGenes)

    def get_hidden_nodes_ids(self):
        return list(self.nodeGenes.keys())[
            self.master_population.INPUT_SIZE : -self.master_population.OUTPUT_SIZE
        ]

    def feed_forward(self, input):
        for i in self.master_population.INPUT_IDS:
            self.nodeGenes[i].state = input[i]
        # first process only hidden nodes:
        # we have to do some magic because we don't know how many hidden nodes, and more importnatly - which ids they have
        for id in self.get_hidden_nodes_ids():
            state = 0
            for connection in self.nodeGenes[id].incoming_connections:
                if connection.enabled:
                    state += self.nodeGenes[connection.from_].state * connection.weight
            self.nodeGenes[id].state = ACTIVATION(state)
        # now only do output nodes, find the argmax
        action = 0
        last_action_activation = -math.inf
        for id in self.master_population.OUTPUT_IDS:
            state = 0
            for connection in self.nodeGenes[id].incoming_connections:
                if connection.enabled:
                    state += self.nodeGenes[connection.from_].state * connection.weight
            state = ACTIVATION(state)
            if state > last_action_activation:
                last_action_activation = state
                action = id
        action -= self.master_population.INPUT_SIZE + self.master_population.MAX_HIDDEN
        return action

    def mutate_weights(self):
        for connection in self.connectionGenes.values():
            if connection.enabled:
                if np.random.random() < COMPLETELY_MUTATE_WEIGHT_CHANCE:
                    connection.weight = np.random.random() * 2 - 1
                else:
                    connection.weight += np.random.normal(scale=0.5) / 10
                connection.weight = clip(connection.weight, -1, 1)

    def recursive_loop_search(self, origin: int, node: int):
        if node == origin:
            return True
        for connection in self.nodeGenes[node].incoming_connections:
            if connection.from_ >= self.master_population.INPUT_SIZE:
                return self.recursive_loop_search(origin, connection.from_)

    def add_connection(self):
        from_candidates = self.master_population.INPUT_IDS + self.get_hidden_nodes_ids()
        np.random.shuffle(from_candidates)
        to_candidates = self.get_hidden_nodes_ids() + self.master_population.OUTPUT_IDS
        np.random.shuffle(to_candidates)
        for from_candidate in from_candidates:
            for to_candidate in to_candidates:
                if from_candidate == to_candidate:
                    continue
                # now that we have node pair, look for existing connection:
                # use szudzik's pairing to deterministically hash them and be able to search quickly
                unique_connection_id = szudzik_pair(from_candidate, to_candidate)
                if unique_connection_id not in self.connectionGenes.keys():
                    recurrent = self.recursive_loop_search(to_candidate, from_candidate)
                    if not recurrent:
                        self.connectionGenes[unique_connection_id] = ConnectionGene(
                            from_candidate, to_candidate, np.random.random() * 2 - 1
                        )
                        self.nodeGenes[to_candidate].incoming_connections.append(
                            self.connectionGenes[unique_connection_id]
                        )
                        return

    def add_node(self):
        if not self.connectionGenes:
            return
        old_connection_unique_id = np.random.choice(list(self.connectionGenes.keys()))
        _fail_safe = 0
        while not self.connectionGenes[old_connection_unique_id].enabled:
            _fail_safe += 1
            if _fail_safe > len(self.connectionGenes) * 500:
                return
            old_connection_unique_id = np.random.choice(
                list(self.connectionGenes.keys())
            )
        old_connection = self.connectionGenes[old_connection_unique_id]
        old_connection.enabled = False
        new_node = NodeGene()
        if (
            old_connection_unique_id
            in self.master_population.pairing_id_to_innovations_map.keys()
        ):
            innovation = self.master_population.pairing_id_to_innovations_map[
                old_connection_unique_id
            ]
        else:
            innovation = self.master_population.innovation_generator.new_id()
            if innovation > self.master_population.MAX_HIDDEN:
                raise ValueError("Innovation number exceeded max hidden nodes count")
            self.master_population.pairing_id_to_innovations_map[
                old_connection_unique_id
            ] = innovation
        self.nodeGenes[innovation] = new_node
        # first half of the old connection is split into one new connection
        # it inherits old connection's weight
        # we want to evolve topology but not introduce too much perturbation and instantly damage fitness
        connection_1_unique_id = szudzik_pair(old_connection.from_, innovation)
        self.connectionGenes[connection_1_unique_id] = ConnectionGene(
            old_connection.from_, innovation, old_connection.weight
        )
        new_node.incoming_connections.append(
            self.connectionGenes[connection_1_unique_id]
        )
        # second half
        # it's weight is initiated to 1 for the reason stated above
        connection_2_unique_id = szudzik_pair(innovation, old_connection.to_)
        self.connectionGenes[connection_2_unique_id] = ConnectionGene(
            innovation, old_connection.to_, 1
        )
        self.nodeGenes[old_connection.to_].incoming_connections.append(
            self.connectionGenes[connection_2_unique_id]
        )
        # we need to insert the new gene BEFORE outputs, right now it's in the end
        # calling sort on this dictionary SHOULD be O(n) since we only have one element out of place, but we should double check this
        self.nodeGenes = OrderedDict(sorted(self.nodeGenes.items(), key=lambda x: x[0]))

    def mutate(self, n_times=1):
        amplify_mutation = MUTATION_BASE_AMPLIFIER * (
            1
            - len(self.connectionGenes)
            / (self.master_population.INPUT_SIZE * self.master_population.OUTPUT_SIZE)
        )
        amplify_mutation = max(amplify_mutation, 1)
        for i in range(n_times):
            if np.random.random() < MUTATE_WEIGHTS_CHANCE:
                self.mutate_weights()
            if np.random.random() < ADD_CONNECTION_CHANCE * amplify_mutation:
                self.add_connection()
            if np.random.random() < ADD_NODE_CHANCE * amplify_mutation:
                self.add_node()
        for gene in self.nodeGenes.values():
            gene.incoming_connections = []
        for connection in self.connectionGenes.values():
            self.nodeGenes[connection.to_].incoming_connections.append(connection)

    # To compare topologies of the genomes we use weight difference and number of disjoint and excess nodes
    # These are computed in parallel for efficiency
    def compare(self, other_genome):
        weight_difference = 0
        shared_connections = 0
        for (id1, connection1) in self.connectionGenes.items():
            for (id2, connection2) in other_genome.connectionGenes.items():
                if id1 == id2:
                    shared_connections += 1
                    weight_difference += abs(connection1.weight - connection2.weight)
        weight_difference = (
            100 if shared_connections == 0 else weight_difference / shared_connections
        )
        disjoint_and_excess = (
            len(self.connectionGenes)
            + len(other_genome.connectionGenes)
            - 2 * shared_connections
        )
        return (
            EXCESS_COEFFICIENT
            * disjoint_and_excess
            / (max(1, len(other_genome.connectionGenes) + len(self.connectionGenes)))
            + WEIGHT_DIFF_COEFFICIENT * weight_difference
        )


class Population:
    class InnovationGenerator:
        def __init__(self, input_size):
            self.innovation = input_size - 1

        def new_id(self):
            self.innovation += 1
            return self.innovation

    def __init__(self, size, input_size, max_hidden_nodes, output_size, logger=None):
        # Configuration
        self.INPUT_SIZE = input_size
        self.MAX_HIDDEN = max_hidden_nodes
        self.OUTPUT_SIZE = output_size
        self.INPUT_IDS = list(range(self.INPUT_SIZE))
        self.OUTPUT_IDS = list(
            range(
                self.INPUT_SIZE + self.MAX_HIDDEN,
                self.INPUT_SIZE + self.MAX_HIDDEN + self.OUTPUT_SIZE,
            )
        )
        self.population_iterator = 0
        self.size = size
        self.population = []
        self.species = []
        self.logger = logger
        self.innovation_generator = self.InnovationGenerator(self.INPUT_SIZE)
        self.pairing_id_to_innovations_map = {}
        for _ in range(size):
            g = Genome.fresh(self)
            g.mutate()
            self.population.append(g)
        self.speciate()
        # Use ids of the connections on which new hidden nodes are created to not create isometric nodes with different innovation numbers
        # This would defeat the purpose of innovation numbers otherwise. This dict is wiped every generation.
        # Map is {connection pairing id : innovation number assigned to the node}

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

    def species_total_fitness(specie):
        total = 0
        for genome in specie:
            total += genome.fitness
        return total

    def choose_parent(self, specie):
        # https://en.wikipedia.org/wiki/Fitness_proportionate_selection
        threshold = np.random.random() * Population.species_total_fitness(specie)
        total = 0
        np.random.shuffle(specie)
        for genome in specie:
            total += genome.fitness
            if total >= threshold:
                return genome

    def best_genome(self):
        max_fitness = -1
        best = None
        for specie in self.species:
            for genome in specie:
                if genome.fitness > max_fitness:
                    best = genome
                    max_fitness = genome.fitness
        self.best = best
        # visualizer.visualize(best)
        return best

    def speciate(self):
        for genome in self.population:
            speciated = False
            for i in range(len(self.species)):
                sample = np.random.choice(self.species[i])
                difference = genome.compare(sample)
                if difference < DIFFERENCE_THRESHOLD:
                    self.species[i].append(genome)
                    speciated = True
                    break
            if not speciated:
                new_species = [genome]
                self.species.append(new_species)

    def forget_dead_species(self):
        self.species = list(filter(lambda x: x != [], self.species))

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
        offspring_nodes = NodeGene.deepcopy_dict(parent1.nodeGenes)
        offspring_connections = {}
        for id1 in parent1.connectionGenes.keys():
            connection_parent = parent1
            enabled = True
            # if another parent has that connection, inherit it with a 0.5 chance
            if id1 in parent2.connectionGenes.keys():
                if np.random.random() < 0.5:
                    connection_parent = parent2
                # if it's disabled in one of the parents, disable it with 0.75 chance
                if (
                    not parent1.connectionGenes[id1].enabled
                    or not parent2.connectionGenes[id1].enabled
                ) and np.random.random() < 0.75:
                    enabled = False
            offspring_connections[id1] = connection_parent.connectionGenes[id1].copy()
            offspring_connections[id1].enabled = enabled
        offspring = Genome(
            offspring_nodes, offspring_connections, parent1.master_population
        )
        return offspring

    def evolve(self):
        average_fitness = 0
        parent_count = 0
        for i in range(len(self.species)):
            self.species[i] = sorted(
                self.species[i], key=lambda x: x.fitness, reverse=True
            )
            self.species[i] = self.species[i][: max(1, len(self.species[i]) // 1)]
            for genome in self.species[i]:
                # we will only use the best half of each specie for mating
                average_fitness += genome.fitness
                parent_count += 1
        self.forget_dead_species()  # do we need this here? TODO
        average_fitness = average_fitness / parent_count
        self.population = []
        left_to_produce = self.size
        while left_to_produce > 0:
            for specie in self.species:
                total_fitness_1 = Population.species_total_fitness(specie)
                if np.random.random() < INTERSPECIES_MATING_CHANCE:
                    other_specie_index = np.random.choice(len(self.species))
                    other_specie = self.species[other_specie_index]
                    total_fitness_2 = Population.species_total_fitness(other_specie)
                    specie_should_produce = (
                        1
                        if (average_fitness == 0)
                        else math.ceil(
                            (total_fitness_1 + total_fitness_2) / average_fitness
                        )
                        * (len(specie) + len(other_specie))
                    )
                else:
                    specie_should_produce = (
                        1
                        if (average_fitness == 0)
                        else math.ceil(total_fitness_1 / average_fitness) * len(specie)
                    )
                left_to_produce -= specie_should_produce
                # if too many individuals have been created, reduce newIndividualsCount to be within the constraints of the population size
                if left_to_produce < 0:
                    specie_should_produce += left_to_produce
                    left_to_produce = 0
                for i in range(specie_should_produce):
                    parent1 = self.choose_parent(specie)
                    parent2 = self.choose_parent(specie)
                    offspring = Population.crossover(parent1, parent2)
                    offspring.mutate()
                    self.population.append(offspring)

        for specie in self.species:
            for genome in specie:
                genome.vestigial = True

        self.speciate()
        for i in range(len(self.species)):
            self.species[i] = list(filter(non_vestigial, self.species[i]))
        self.forget_dead_species()
        if self.logger:
            self.logger.info("Average fitness reached: {}".format(average_fitness))
        self.pairing_id_to_innovations_map = {}
