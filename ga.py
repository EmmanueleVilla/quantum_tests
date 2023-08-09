import numpy as np
from qiskit import Aer, QuantumCircuit, transpile, assemble

from build_circuit import build_circuit
from dataset import create_dataset


def initialize_population(pop_size, num_theta):
    population = []
    for _ in range(pop_size):
        individual = np.random.uniform(-2 * np.pi, 2 * np.pi, num_theta)
        population.append(individual)
    return population


def create_circuit():
    return build_circuit()


def fitness_function(theta, features, labels):
    backend = Aer.get_backend('qasm_simulator')

    accurate_predictions = 0
    total_samples = len(features)

    for feature, label in zip(features, labels):
        qc = QuantumCircuit(12, 1)

        for i, val in enumerate(feature):
            if val == 1:
                qc.x(i)

        qc += create_circuit()

        qc.measure(12, 0)
        qc.bind_parameters(theta)

        transpiled_circuit = transpile(qc, backend)
        qobj = assemble(transpiled_circuit, shots=1)
        results = backend.run(qobj).result()
        counts = results.get_counts()

        if '1' in counts:
            predicted_label = 1
        else:
            predicted_label = 0

        if predicted_label == label:
            accurate_predictions += 1

    accuracy = accurate_predictions / total_samples
    loss = 1 - accuracy
    return accuracy, loss


def tournament_selection(population, fitness_scores, tournament_size):
    tournament_indices = np.random.choice(len(population), size=tournament_size, replace=False)
    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
    winner_index = tournament_indices[np.argmax(tournament_fitness)]
    return winner_index


def two_point_crossover(parent1, parent2):
    point1 = np.random.randint(0, len(parent1))
    point2 = np.random.randint(point1 + 1, len(parent1))
    child = np.copy(parent1)
    child[point1:point2] = parent2[point1:point2]
    return child


def mutation(child, mutation_rate):
    for i in range(len(child)):
        if np.random.random() < mutation_rate:
            if np.random.random() < 0.5:
                child[i] += np.random.uniform(-0.1, 0.1)
            else:
                child[i] = np.random.uniform(-2 * np.pi, 2 * np.pi)
    return child


def save_population(population, filename):
    with open(filename, "w") as f:
        for individual in population:
            f.write(" ".join(map(str, individual)) + "\n")


def load_population(filename):
    population = []
    with open(filename, "r") as f:
        for line in f:
            individual = list(map(float, line.strip().split()))
            population.append(individual)
    return population


def main():

    population_size = 100
    num_theta = 93
    generations = 50
    mutation_rate = 0.1
    tournament_size = 10

    train_features, train_labels, test_features, test_labels = create_dataset()

    population = initialize_population(population_size, num_theta)

    fitness_scores = np.zeros(len(population))

    for generation in tqdm(range(generations), desc="Generations", unit="generation"):
        fitness_scores = [fitness_function(individual, train_features, train_labels) for individual in population]
        new_population = []

        for _ in range(population_size):
            parent1_index = tournament_selection(population, fitness_scores, tournament_size)
            parent2_index = tournament_selection(population, fitness_scores, tournament_size)

            parent1 = population[parent1_index]
            parent2 = population[parent2_index]

            child = two_point_crossover(parent1, parent2)
            child = mutation(child, mutation_rate)

            new_population.append(child)

        population = new_population

    best_individual = population[np.argmax(fitness_scores)]
    best_fitness = max(fitness_scores)
    print("Miglior individuo:", best_individual)
    print("Miglior punteggio di fitness:", best_fitness)


main()
