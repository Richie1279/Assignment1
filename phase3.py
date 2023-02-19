import random
import matplotlib.pyplot as plt
random.seed(101) 
# important to use same seed for repeatability in testing


class Chromosome:
    def __init__(self, _size):  
        self.fitness = 0  
        self.genes = []  # a list to represent the sequence of 1s and 0s
        self.chromosome_size = _size 

        # within the constructor itself, we randomly create the specified number of genes
        # and append those values to the genes list
        for _ in range(self.chromosome_size):
            self.genes.append(random.randint(0, 1))

    # a function to calculate the fitness of the chromosome
    # in this case, the aside from the normal fittest function of all 1's. 
    # We have a better solution whereby a fitness of 60 can be achieved if the 
    # every bit is 0 i.e the chromosome sum is 0
    def compute_fitness(self):
        count = sum(self.genes)
        if count == 0:
            self.fitness = 2 * self.chromosome_size
        else:
            self.fitness = count

    # a function to mutate a randomly selected gene position
    # when a index is passes, we just flip that particular gene value
    def mutate(self, gene_index):
        self.genes[gene_index] = 1 - self.genes[gene_index]


class GeneticAlgorithm:
    # constructor 
    def __init__(self, _pop_size, _max_gen, _chromosome_size, _mutation_rate ):
        self.population = []  # to represent the actual population of chromosomes
        # to represent the selected chromosomes that can mate to produce children
        self.mating_pool = []
        self.best = None  # to hold the current best child in the generation
        self.population_size = _pop_size  # number of chromosomes within the population
        # number of genes within the chromosome
        self.chromosome_size = _chromosome_size
        # rate at which mutation happens. This should be a very low probability
        self.mutation_rate = _mutation_rate
        self.generation = 0  
        self.avg_fitness = []  # average fitness of each generation
        # max number of generations allowed to search for the solution chromosome
        self.max_generation = _max_gen
        self.init_population() # initialise population 

    
    def init_population(self):
        for _ in range(self.population_size):
            # create an object of the class
            chromosome = Chromosome(self.chromosome_size)
            # since the creation of the chromosome object triggers the creation of genes
            # at random  we append the chromosome to the population
            self.population.append(chromosome)
        # this function is called only on the initial population
        # so we assume the first child within the population to be the best one
        self.best = self.population[0]

    # a function to update the mating pool
    # since we need population size number of chromosomes to mate,
    # we simply loop that many times. within the loop, we randomly select 2 chromosomes
    # from the population and compare the fitness value of them (tournament)
    # the fittest chromosome gets added to the mating pool
    def update_mating_pool(self):
        self.mating_pool = []
        for _ in range(self.population_size):
            chromosome_1 = self.population[random.randint(
                0, self.population_size - 1)]
            chromosome_2 = self.population[random.randint(
                0, self.population_size - 1)]

            if chromosome_1.fitness > chromosome_2.fitness:
                self.mating_pool.append(chromosome_1)
            else:
                self.mating_pool.append(chromosome_2)

    # a function to select 2 parents from the mating pool at random
    def random_selection(self):
        parent_1 = self.mating_pool[random.randint(
            0, self.population_size - 1)]
        parent_2 = self.mating_pool[random.randint(
            0, self.population_size - 1)]
        return parent_1, parent_2

    # a function to perform the actual crossover
    # this function accepts 2 parents and return the newly created child
    def crossover(self, parent_1, parent_2):
        # initialize a child chromosome
        child = Chromosome(self.chromosome_size)

        # select a random point that is in between 1 and chromosome size - 2
        # 1 and size - 2 is because we atleast need one gene at the front or back of the random
        # point to perform the crossover
        random_point = random.randint(1, self.chromosome_size - 2)

        # now we copy the first part of the first parent and second part of the second parent
        # to the genes of the initialized child chromosome
        child.genes = parent_1.genes[:random_point] + \
            parent_2.genes[random_point:]

        return child

    # a function to perform the mutation
    def mutation(self, child):
        # we go from the start to end of the child's genes and every time we
        # see a gene, we generate a random number. If that number is less than the mutation probability,
        # we flip that gene
        for i in range(self.chromosome_size):
            if random.random() < self.mutation_rate:
                child.mutate(i)

    # a function to generate the new set of children for the next generation
    def new_generation(self):
        fitness_sum = 0  # a variable to hold the sum of fitness values of each child generated
        for i in range(self.population_size):
            # select 2 parents at random
            parent_1, parent_2 = self.random_selection()
            # perform the crossover and create a new child
            child = self.crossover(parent_1, parent_2)
            # perform the mutation
            self.mutation(child)
            # replace the current chromosome within the population with the newly created child
            # remember we select the parents to mate from the mating pool.
            # therefore, it is ok to replace the population here as it is not used at this point
            self.population[i] = child

            # compute the fitness of the new child
            child.compute_fitness()
            # add the fitness value to the sum
            fitness_sum += child.fitness

            # if the current child is fit than the best child, replace the best child and print some details of it
            if child.fitness > self.best.fitness:
                self.best = child
                print("\nGeneration:", self.generation)
                print("Chromosome:", self.best.genes)
                print("Fitness: ", self.best.fitness)
                print("=========================")
        # in the end, outside the loop, we divide the sum of fitness with the population size to
        # get the average fitness of this generation
        generation_fitness_average = fitness_sum / self.population_size
        self.avg_fitness.append(generation_fitness_average)

    # a function to actually perform the solution search
    def search(self):
        self.generation = 0  
        # the exit condition in our case are
        # 1. the current generation exceed the max generations allowed
        # 2. the best child has a fitness of 30
        while self.generation < self.max_generation and self.best.fitness < 30:
            # calculate the fitness of each chromosome in the population
            for chromosome in self.population:
                chromosome.compute_fitness()
            # update the mating pool
            self.update_mating_pool()
            # generate new child chromosomes to replace the current ones from the population
            self.new_generation()
            # iterate the generation count
            self.generation += 1
        print("Generation:", self.generation)

        # we plot the graph using average fitness list
        plt.plot(self.avg_fitness)
        plt.xlabel('Generations')
        plt.ylabel('Average fitness')
        plt.title('GA')
        plt.show()


# create an object of the GeneticAlgorithm class passing in values for
# the population, max generations, chromosome size, and mutation rate
onemax = GeneticAlgorithm(100, 70, 30, 0.05)
onemax.search()  