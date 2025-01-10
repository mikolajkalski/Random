import numpy as np
import matplotlib.pyplot as plt


#1. Przygotowanie zbioru danych dla operatora XOR
# Dane wejściowe (X) i oczekiwane wyniki (y) dla operatora XOR
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([[0], [1], [1], [0]])  # Wyjścia oczekiwane



"""Co robi kod:
X definiuje dane wejściowe jako kombinacje dwóch bitów.
y to oczekiwane wyjścia dla danych wejściowych zgodnie z operatorem XOR.
"""

#2. Inicjalizacja parametrów sieci
# Liczba neuronów
n_input = 2  # Warstwa wejściowa
n_output = 1  # Warstwa wyjściowa

# Inicjalizacja wag i biasów z rozkładu normalnego
np.random.seed(42)  # Dla powtarzalności wyników
weights = np.random.randn(n_input, n_output)  # Wagi (2x1)
biases = np.random.randn(n_output)  # Bias (1)

"""Co robi kod:
weights to macierz wag dla każdego wejścia (2 wejścia -> 1 neuron wyjściowy).
biases to wektor biasów, jeden dla każdego neuronu wyjściowego.
"""

#2. Inicjalizacja parametrów sieci

# Liczba neuronów
n_input = 2  # Warstwa wejściowa
n_output = 1  # Warstwa wyjściowa

# Inicjalizacja wag i biasów z rozkładu normalnego
np.random.seed(42)  # Dla powtarzalności wyników
weights = np.random.randn(n_input, n_output)  # Wagi (2x1)
biases = np.random.randn(n_output)  # Bias (1)
"""Co robi kod:
weights to macierz wag dla każdego wejścia (2 wejścia -> 1 neuron wyjściowy).
biases to wektor biasów, jeden dla każdego neuronu wyjściowego."""

#3. Implementacja funkcji aktywacji

# Funkcja sigmoidalna
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Pochodna funkcji sigmoidalnej
def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)
"""Co robi kod:
sigmoid(x) oblicza wartość funkcji sigmoidalnej, używanej do aktywacji neuronów.
sigmoid_derivative(x) oblicza pochodną funkcji sigmoidalnej, potrzebną do propagacji wstecznej.
"""
#4. Propagacja w przód

def forward_propagation(X, weights, biases):
    z = np.dot(X, weights) + biases  # Obliczenie sumy ważonej
    y_pred = sigmoid(z)  # Zastosowanie funkcji aktywacji
    return y_pred, z
"""Co robi kod:
z to suma ważona obliczona jako z=𝑋⋅𝑤+b
y_pred to wyjście po zastosowaniu funkcji aktywacji."""

#5. Propagacja wstecz

def backward_propagation(X, y, y_pred, z):
    error = y_pred - y  # Obliczenie błędu
    d_weights = np.dot(X.T, error * sigmoid_derivative(z))  # Gradient wag
    d_biases = np.sum(error * sigmoid_derivative(z), axis=0)  # Gradient biasów
    return d_weights, d_biases
"""Co robi kod:
error to różnica między przewidywanymi a rzeczywistymi wynikami.
d_weights i d_biases to gradienty potrzebne do aktualizacji wag i biasów."""

#6. Aktualizacja wag i biasów

def update_parameters(weights, biases, d_weights, d_biases, learning_rate):
    weights -= learning_rate * d_weights  # Aktualizacja wag
    biases -= learning_rate * d_biases  # Aktualizacja biasów
    return weights, biases
"""Co robi kod:
Używa gradientów i współczynnika uczenia (learning_rate) do modyfikacji wag i biasów."""
#7. Pętla uczenia

# Parametry algorytmu
learning_rate = 0.1
n_epochs = 10000  # Liczba epok

mse_history = []

for epoch in range(n_epochs):
    # Propagacja w przód
    y_pred, z = forward_propagation(X, weights, biases)
    
    # Obliczanie błędu MSE
    mse = np.mean((y - y_pred) ** 2)
    mse_history.append(mse)
    
    # Propagacja wstecz
    d_weights, d_biases = backward_propagation(X, y, y_pred, z)
    
    # Aktualizacja parametrów
    weights, biases = update_parameters(weights, biases, d_weights, d_biases, learning_rate)
    
    # Monitorowanie postępu
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, MSE: {mse:.4f}")
"""Co robi kod:
W każdej epoce:
Oblicza propagację w przód.
Liczy średniokwadratowy błąd (MSE).
Oblicza gradienty w propagacji wstecz.
Aktualizuje wagi i biasy.
Monitoruje błąd co 1000 epok."""

#8. Wizualizacja wyników


# Wykres MSE w czasie
plt.plot(mse_history)
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('MSE During Training')
plt.show()

# Testowanie sieci
y_pred, _ = forward_propagation(X, weights, biases)
print("Przewidywane wyjścia:", np.round(y_pred).flatten())
print("Oczekiwane wyjścia:", y.flatten())
"""Co robi kod:
Rysuje wykres MSE w trakcie uczenia.
Pokazuje przewidywane wyniki sieci w porównaniu z oczekiwanymi.
Wyniki i wnioski
Działanie sieci: Sieć powinna poprawnie nauczyć się operatora XOR, osiągając niski błąd MSE i poprawne przewidywania.
Współczynnik uczenia: Dostosowanie learning_rate wpływa na szybkość uczenia.
Liczba epok: Więcej epok pozwala na dokładniejsze dopasowanie"""