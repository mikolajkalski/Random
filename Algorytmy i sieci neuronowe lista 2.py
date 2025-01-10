import random

def tournament_selection(population, fitness_values, tournament_size=3):
 
    tournament = random.sample(list(zip(population, fitness_values)), tournament_size)
    winner = max(tournament, key=lambda x: x[1])
    return winner[0]


population = [[1, 0, 1, 0, 1], [0, 1, 1, 0, 0], [1, 1, 0, 1, 0], [0, 0, 1, 1, 1]]
fitness_values = [30, 20, 25, 15]
selected = tournament_selection(population, fitness_values)
print("Selected:", selected)
#Co robi kod:
#Losowy wybór: Losowo wybiera tournament_size osobników z populacji.
#Porównanie przystosowania: Porównuje ich wartości funkcji fitness.
#Zwraca najlepszego: Wybiera osobnika o najwyższej wartości fitness.


#2 metoda ruletki
def roulette_wheel_selection(population, fitness_values):
    
    total_fitness = sum(fitness_values)
    probabilities = [f / total_fitness for f in fitness_values]

    rand = random.random()
    cumulative_probability = 0
    for individual, probability in zip(population, probabilities):
        cumulative_probability += probability
        if rand <= cumulative_probability:
            return individual


selected = roulette_wheel_selection(population, fitness_values)
print("Selected individual:", selected)

#Co robi kod:
#Oblicza prawdopodobieństwa selekcji: Na podstawie wartości fitness każdego osobnika.
#Generuje losowy wybór: Wybiera osobnika zgodnie z prawdopodobieństwem (imituje obrót kołem ruletki).
#Zwraca wybranego osobnika.

#3. mutacja punktowa
def point_mutation(chromosome, mutation_rate=0.1):
    mutated = chromosome[:]
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            mutated[i] = 1 - mutated[i]  
    return mutated


chromosome = [1, 0, 1, 0, 1]
mutated_chromosome = point_mutation(chromosome, mutation_rate=0.2)
print("Original chromosome:", chromosome)
print("Mutated chromosome:", mutated_chromosome)

#Co robi kod:
#Klonowanie chromosomu: Tworzy kopię, aby oryginalny chromosom pozostał niezmieniony.
#Iteracja po genach: Przechodzi przez każdy gen w chromosomie.
#Mutacja: Z prawdopodobieństwem mutation_rate odwraca gen (0 → 1 lub 1 → 0).


#4. porownanie metod
def run_simulation(selection_method, population, fitness_values, iterations=10):
     for _ in range(iterations):
        selected = selection_method(population, fitness_values)
        print("Selected individual:", selected)

#test TS
print("TS:")
run_simulation(lambda pop, fit: tournament_selection(pop, fit, tournament_size=3), 
               population, fitness_values)

#test Ruletki
print("\nRuleta cyk cyk:")
run_simulation(roulette_wheel_selection, population, fitness_values)

#Co robi kod:
#Wielokrotne iteracje: Wybiera osobników wielokrotnie przy użyciu określonej metody selekcji.
#Porównanie wyników: Pozwala zaobserwować, które osobniki są częściej wybierane.


#5. wplyw mutacji
def genetic_diversity(population):
    unique_individuals = len(set(tuple(chromosome) for chromosome in population))
    return unique_individuals / len(population)


mutated_population = [point_mutation(individual, mutation_rate=0.1) for individual in population]
diversity = genetic_diversity(mutated_population)
print("Genetic Diversity po mutacji:", diversity)

#Co robi kod:
#Tworzy zbiór unikalnych chromosomów: Zamienia chromosomy na krotki (tuples), aby były hashowalne.
#Oblicza różnorodność: Dzieli liczbę unikalnych chromosomów przez całkowitą liczbę osobników.