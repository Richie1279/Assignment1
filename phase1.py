# we will need 3 main functions
# 1. population initializing function
# 2. fitness function
# 3. crossover function

import random
random.seed(101)

population_size = 100
mutation_probability = 0.1
chromosome_size = 30


def initialize_population():
    population = []
    population_fitness_sum = 0

    for i in range(population_size):
        chromosome = []
        for _ in range(chromosome_size):
            gene = random.choices([0, 1])
            chromosome.append(gene[0])
        fitness = fitness_function(chromosome)
        population_fitness_sum += fitness
        population.append(chromosome)
    return population, population_fitness_sum


def fitness_function(chromosome):
    return sum(chromosome)


def crossover(parent_1, parent_2):
    random_point = random.randint(1, chromosome_size - 2)
    child_1 = parent_1[:random_point] + parent_2[random_point:]
    child_2 = parent_2[:random_point] + parent_1[random_point:]
    return [child_1, child_2]


population, population_fitness_sum = initialize_population()

new_population = []
parents = random.choices(population, k=2)
children = crossover(parents[0], parents[1])
