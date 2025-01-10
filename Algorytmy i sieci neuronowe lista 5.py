import numpy as np

#1.modyfikacja procesu generowania populacji poczatkowej

# Funkcja fitness
def fitness_function(individual):
    return np.sum(np.square(individual))

# Losowa inicjalizacja według rozkładu jednostajnego
def initialize_population_uniform(size, n_genes, low=-10, high=10):
    # Tworzy populację losowaną według rozkładu jednostajnego
    return np.random.uniform(low, high, (size, n_genes))

# Losowa inicjalizacja według rozkładu normalnego
def initialize_population_normal(size, n_genes, mean=0, std=5):
    # Tworzy populację losowaną według rozkładu normalnego
    return np.random.normal(mean, std, (size, n_genes))

# Test porównawczy różnych metod inicjalizacji
def test_initialization_methods(pop_size, n_genes):
    # Tworzy populacje różnymi metodami
    uniform_population = initialize_population_uniform(pop_size, n_genes)
    normal_population = initialize_population_normal(pop_size, n_genes)

    # Ewaluacja populacji
    uniform_fitness = [fitness_function(ind) for ind in uniform_population]
    normal_fitness = [fitness_function(ind) for ind in normal_population]

    # Statystyki
    stats = {
        "uniform": {
            "mean": np.mean(uniform_fitness),
            "max": np.max(uniform_fitness),
            "min": np.min(uniform_fitness),
        },
        "normal": {
            "mean": np.mean(normal_fitness),
            "max": np.max(normal_fitness),
            "min": np.min(normal_fitness),
        },
    }
    return stats

#2.impementacja nowych metod

# Selekcja rankingowa
def rank_selection(population, fitnesses, rank_bias=1.5):
    # Sortuje populację według przystosowania
    sorted_indices = np.argsort(fitnesses)[::-1]
    sorted_population = population[sorted_indices]
    ranks = np.arange(1, len(population) + 1)
    probabilities = (1 / ranks**rank_bias) / np.sum(1 / ranks**rank_bias)
    # Wybiera osobnika na podstawie rankingu
    selected_index = np.random.choice(len(population), p=probabilities)
    return sorted_population[selected_index]

# Selekcja ruletkowa z adaptacyjnym prawdopodobieństwem
def roulette_wheel_selection(population, fitnesses):
    total_fitness = np.sum(fitnesses)
    probabilities = fitnesses / total_fitness
    # Wybór osobnika na podstawie prawdopodobieństwa proporcjonalnego do fitness
    selected_index = np.random.choice(len(population), p=probabilities)
    return population[selected_index]

# Selekcja turniejowa z różną liczbą uczestników
def tournament_selection_with_size(population, fitnesses, tournament_size):
    participants = np.random.choice(len(population), tournament_size, replace=False)
    best = participants[np.argmax(fitnesses[participants])]
    return population[best]

# Testowanie metod selekcji
def test_selection_methods(population, fitnesses):
    selected_rank = rank_selection(population, fitnesses)
    selected_roulette = roulette_wheel_selection(population, fitnesses)
    selected_tournament = tournament_selection_with_size(population, fitnesses, 3)
    return {
        "rank": selected_rank,
        "roulette": selected_roulette,
        "tournament": selected_tournament,
    }