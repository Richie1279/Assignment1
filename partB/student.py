import pandas as pd
import matplotlib.pyplot as plt
import random
random.seed(101)


# a function to read the excel files
# we use pandas library to read excel files as it is one of
# the most common and easiest methods to read different file formats
def read_files_and_generate_matrix():
    # first we specify the file names
    lecturers_file = pd.read_excel('Supervisors.xlsx')
    students_file =  pd.read_excel('Student-choices.xls')

    # using pandas read_excel method, we read the excel files.
    # since there are no heading row within the excel file, we specify that using the header parameter
    lecturers = pd.read_excel(lecturers_file, header=None)
    students = pd.read_excel(students_file, header=None)

    # a list to store the list of student preferences. This will be a list of lists in the end
    student_preference = []
    # a list to store the lecturer capacity. index of the list will specify the lecturer
    lecturer_capacity = []

    # for each row in lecturers dataframe, we append the value to the list
    for _, row in lecturers.iterrows():
        lecturer_capacity.append(row[1])

    # for each row, we go through all the columns and store it in a different list
    # in the end, we append the list to the students preference list
    for _, row in students.iterrows():
        pref_row = []
        for i in range(1, len(row)):
            pref_row.append(row[i])
        student_preference.append(pref_row)

    return lecturer_capacity, student_preference


class Chromosome:
    def __init__(self, _num_of_students, _lecturer_capacity):
        self.fitness = 0
        self.genes = []
        self.chromosome_size = _num_of_students
        self.init_chromosome(_lecturer_capacity)

    # a function to initialize chromosomes
    # in our case, each index of the gene within the chromosome represents a student
    # and each gene represents the lecturer assigned to the corresponding student (index)
    # instead of randomly assigning a student to a lecturer,
    # this function accepts the lecturer capacity list which will be used to see if the assignment
    # of a student exceeds the capacity of the lecturer
    def init_chromosome(self, lecturer_capacity):
        num_of_lecturers = len(lecturer_capacity)

        # a list to keep track of lecturers total assignments. we initialize with all zeros and each index
        # represents a lecturer
        lecturer_assignment = [0] * num_of_lecturers
        # until the chromosome size is not met, we loop
        while len(self.genes) < self.chromosome_size:
            # randomly generate a lecturer between zero and max lecturers available (lecturer index)
            random_lecturer = random.randint(0, num_of_lecturers - 1)
            # check if assignment count exceeded capacity
            if (lecturer_assignment[random_lecturer] < lecturer_capacity[random_lecturer]):
                # append the lecturer to the gene
                self.genes.append(random_lecturer)
                # update assignment count list
                lecturer_assignment[random_lecturer] += 1

    # function to compute the fitness. Fitness is the percentage of preference match of student
    def compute_fitness(self, student_preference):
        assignment_sum = 0  # a variable to keep track of the current assignment score
        for i in range(self.chromosome_size):
            student = i  # each index of the chromosome represents the student
            # each value within the gene represents the lecturer
            lecturer = self.genes[i]
            # we check the student preference table to find the current preference score of the assignment and add it to the variable
            assignment_sum += student_preference[student][lecturer]
        # to scale the value between 0 and 1, we divide the number of students by the calculated score
        self.fitness = self.chromosome_size / assignment_sum

    # since this is a permutation problem, mutation operation will be a bit different from the previous one
    # we swap the assignments of students and lecturers by randomly swapping the values of genes
    def mutate(self, index_1, index_2):
        self.genes[index_1], self.genes[index_2] = self.genes[index_2], self.genes[index_1]


class GeneticAlgorithm:
    def __init__(self, _pop_size, _max_gen, _chromosome_size, _mutation_rate, _lecturer_capacity, _student_preference):
        self.population = []
        self.mating_pool = []
        self.best = None
        self.population_size = _pop_size
        self.chromosome_size = _chromosome_size
        self.mutation_rate = _mutation_rate
        self.generation = 0
        self.avg_fitness = []
        self.max_generation = _max_gen
        self.lecturer_capacity = _lecturer_capacity
        self.student_preference = _student_preference
        self.init_population()

    def init_population(self):
        for _ in range(self.population_size):
            chromosome = Chromosome(
                self.chromosome_size, self.lecturer_capacity)
            self.population.append(chromosome)
        self.best = self.population[0]

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

    def random_selection(self):
        parent_1 = self.mating_pool[random.randint(
            0, self.population_size - 1)]
        parent_2 = self.mating_pool[random.randint(
            0, self.population_size - 1)]
        return parent_1, parent_2

    # crossover cannot be one point as this is a permutation problem
    # for the moment, we are selecting the best parent as the child
    def crossover(self, parent_1, parent_2):
        child = Chromosome(self.chromosome_size, self.lecturer_capacity)

        if parent_1.fitness > parent_2.fitness:
            child.genes = parent_1.genes[:]
        else:
            child.genes = parent_2.genes[:]

        return child

    # whenever we decide to mutate, we randomly pick 2 indices from the genes and swap them
    def mutation(self, child):
        for i in range(self.chromosome_size):
            if random.random() < self.mutation_rate:
                index_1 = random.randint(0, self.chromosome_size - 1)
                index_2 = random.randint(0, self.chromosome_size - 1)
                child.mutate(index_1, index_2)

    def new_generation(self):
        fitness_sum = 0
        for i in range(self.population_size):
            parent_1, parent_2 = self.random_selection()
            child = self.crossover(parent_1, parent_2)
            self.mutation(child)
            self.population[i] = child

            child.compute_fitness(self.student_preference)
            fitness_sum += child.fitness

            if child.fitness > self.best.fitness:
                self.best = child
                print("\nGeneration:", self.generation)
                print("Chromosome:", self.best.genes)
                print("Fitness: ", self.best.fitness)
                print("=========================")
        generation_fitness_average = fitness_sum / self.population_size
        self.avg_fitness.append(generation_fitness_average)

    def search(self):
        self.generation = 0
        while self.generation < self.max_generation and self.best.fitness < 1:
            for chromosome in self.population:
                chromosome.compute_fitness(self.student_preference)
            self.update_mating_pool()
            self.new_generation()
            self.generation += 1
        print("Generation:", self.generation)

        plt.plot(self.avg_fitness)
        plt.xlabel('Generations')
        plt.ylabel('Average fitness')
        plt.title('GA')
        plt.show()


lecturer_capacity, student_preference = read_files_and_generate_matrix()
pop_size = 50
max_gen = 5000
chromosome_size = len(student_preference)
mutation_rate = 0.01

ga = GeneticAlgorithm(
    pop_size,
    max_gen,
    chromosome_size,
    mutation_rate,
    lecturer_capacity,
    student_preference
)
ga.search()
