
#https://drive.google.com/file/d/1LIDgekQWFsTtdlFG1fZrRtQKo6hvASi_/view
#zad. prakt.

import random

values = [10, 20, 30, 40, 50]
weight = [1, 2, 3, 8, 7]

chromosome = [1, 0, 1, 0, 1]

def generate_random_popularion(population_size, chromosome_lenght):
    population = []
    for _ in range(population_size):
        chromosome = [random.randint(0, 1)for _ in range(chromosome_lenght)]
        population.append(chromosome)
    return population

#przykład
#population_size = 10
#chromosome_length = 5 (liczba itemow w chromie)
#population = generate_random_population(population_size, chromosome_lenght)
#print("")
#for chromosome in population
#   print(chromosome)


def fitness_func(chromosome, values, weights, max_capacity):
#knapsack problem calc
#chomosome list, values list, weights list, max_cap int
    total_value = sum(value for gene, value in zip(chromosome, values,) if gene == 1)
    total_weight = sum(weight for gene, weight in zip(chromosome, weights) if gene == 1)

#jezeli wartosc jest over cap
    if total_weight > max_capacity:
        penalty = (total_weight - max_capacity) * 2
        return total_value - penalty

    return total_value

#przykład
values = [10, 20, 30, 40, 50]
weight = [1, 2, 3, 4, 5]
max_capacity = 10

chromosome = [1, 0, 1, 0, 1]  
fitness = fitness_func(chromosome, values, weight, max_capacity)
print("Fitness:", fitness)

#wyjasnienie: suma wartosci przedmiotow w placku,
#  jezeli suma < -> kara, proporcjonalna do overcap