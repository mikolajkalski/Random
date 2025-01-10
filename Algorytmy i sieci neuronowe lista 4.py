import numpy as np

# Funkcja fitness, np. optymalizacja maksymalizacji kwadratu
def fitness_function(individual):
    # Oblicza wartość funkcji celu (tu suma kwadratów)
    return np.sum(np.square(individual))

# Inicjalizacja populacji
def initialize_population(size, n_genes):
    # Tworzy losową populację o zadanym rozmiarze i liczbie genów
    return np.random.uniform(-10, 10, (size, n_genes))

# Ewaluacja populacji
def evaluate_population(population):
    # Oblicza wartość funkcji przystosowania dla każdego osobnika
    return np.array([fitness_function(ind) for ind in population])

# Funkcja selekcji turniejowej
def tournament_selection(population, fitnesses, tournament_size):
    # Wybór uczestników turnieju
    participants = np.random.choice(len(population), tournament_size, replace=False)
    # Zwraca osobnika o najlepszym fitness z wybranych
    best = participants[np.argmax(fitnesses[participants])]
    return population[best]

# Krzyżowanie jednopunktowe
def one_point_crossover(parent1, parent2):
    # Wybiera losowy punkt podziału
    point = np.random.randint(1, len(parent1) - 1)
    # Tworzy dzieci poprzez wymianę genów
    child1 = np.concatenate((parent1[:point], parent2[point:]))
    child2 = np.concatenate((parent2[:point], parent1[point:]))
    return child1, child2

# Mutacja
def mutate(individual, mutation_rate):
    # Dla każdego genu wykonuje mutację z określonym prawdopodobieństwem
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            # Mutacja poprzez dodanie losowej wartości
            individual[i] += np.random.uniform(-1, 1)
    return individual

# Algorytm genetyczny
def genetic_algorithm(pop_size, n_genes, n_generations, mutation_rate, tournament_size):
    # Inicjalizuje populację
    population = initialize_population(pop_size, n_genes)
    # Historia statystyk
    stats = {"best": [], "worst": [], "mean": []}

    for generation in range(n_generations):
        # Ewaluacja populacji
        fitnesses = evaluate_population(population)

        # Zapisuje statystyki
        stats["best"].append(np.max(fitnesses))
        stats["worst"].append(np.min(fitnesses))
        stats["mean"].append(np.mean(fitnesses))

        # Tworzenie nowej populacji
        new_population = []
        while len(new_population) < pop_size:
            # Selekcja
            parent1 = tournament_selection(population, fitnesses, tournament_size)
            parent2 = tournament_selection(population, fitnesses, tournament_size)
            # Krzyżowanie
            child1, child2 = one_point_crossover(parent1, parent2)
            # Mutacja
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            # Dodanie dzieci do nowej populacji
            new_population.extend([child1, child2])

        # Aktualizacja populacji
        population = np.array(new_population[:pop_size])

    return stats


    #Zadania NIEZREALIZOWANE:
"""Badanie wpływu rozmiaru populacji:

Kod nie zawiera testów dla różnych rozmiarów populacji (np. 2, 6, 10, 50, 200) i porównania statystyk (najlepszy/najgorszy/średni fitness).
Wymagana jest modyfikacja głównej funkcji, aby umożliwić testowanie.
Porównanie metod selekcji:

Implementacja zawiera jedynie selekcję turniejową. Brakuje np. selekcji ruletkowej czy rankingowej.
Alternatywne operatory krzyżowania:

Kod zawiera tylko krzyżowanie jednopunktowe. Brakuje np. krzyżowania dwupunktowego czy jednorodnego.
Wpływ prawdopodobieństwa mutacji:

Brak eksperymentów z różnymi wartościami prawdopodobieństwa mutacji.
Monitorowanie statystyk populacji:

Kod rejestruje tylko najlepszy, najgorszy i średni fitness, ale brak szczegółowej analizy rozkładu fitness w populacji.
Wizualizacje wyników:

Nie ma żadnych wykresów ani narzędzi do wizualizacji statystyk pokoleniowych.
Sformułowanie wniosków:

Kod nie zawiera mechanizmu raportowania ani analizy wniosków.