from datetime import datetime

import numpy as np
from qiskit import Aer, QuantumCircuit, transpile, assemble
from tqdm import tqdm

from build_circuit import build_circuit
from dataset import create_dataset

if __name__ == "__main__":
    def initialize_population(pop_size, num_theta):
        population = []
        for _ in range(pop_size):
            individual = np.random.uniform(0, np.pi / 4, num_theta)
            population.append(individual)
        return population


    def create_circuit():
        return build_circuit()


    def fitness_function(theta, features, labels):
        backend = Aer.get_backend('qasm_simulator')

        accurate_predictions = 0

        qc = QuantumCircuit(13, 13)

        qnn = create_circuit()
        qnn.assign_parameters(theta, inplace=True)

        qc = qc.compose(qnn.copy("QNN"))
        qc.barrier()
        qc.cx(11, 12)
        qc.barrier()
        qc = qc.compose(qnn.copy("Inv QNN").inverse())
        qc.barrier()
        qc.measure(range(12), range(12))
        qc.measure(12, 12)

        transpiled_circuit = transpile(qc, backend)
        results = backend.run(transpiled_circuit, shots=100000).result()
        counts = results.get_counts()

        ordered_count = sorted(counts.items(), key=lambda x: x[1], reverse=True)

        tot = 0

        for i in range(0, len(features)):
            feature = features[i]
            label = labels[i]

            for count in ordered_count:
                graph = count[0][0:12]
                prediction = count[0][12]

                if graph == feature:
                    tot += 1
                    if str(label) == "1" and prediction == "1":
                        accurate_predictions += 1
                    if str(label) == "-1" and prediction == "0":
                        accurate_predictions += 1
                    break

        accuracy = accurate_predictions / tot
        return accuracy


    def tournament_selection(population, fitness_scores, tournament_size):
        tournament_indices = np.random.choice(len(population), size=tournament_size, replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_index = np.argmax(tournament_fitness)
        return winner_index


    def two_point_crossover(parent1, parent2):
        point1 = np.random.randint(0, len(parent1) - 10)
        point2 = np.random.randint(point1 + 1, len(parent1))
        child = np.copy(parent1)
        child[point1:point2] = parent2[point1:point2]
        return child


    def mutation(child, mutation_rate):
        for i in range(len(child)):
            if np.random.random() < mutation_rate:
                child[i] += np.random.uniform(-0.05, 0.05)
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

        population_size = 50
        num_theta = 93
        generations = 50
        mutation_rate = 0.1
        tournament_size = 8

        train_features, train_labels, test_features, test_labels = create_dataset(100)

        population = initialize_population(population_size, num_theta)

        fitness_scores = np.zeros(len(population))

        # Print the starting time and hour
        print("Starting time:", datetime.now().time())

        features_graph = [''.join(str(x) for x in row) for row in train_features]

        for generation in tqdm(range(generations), desc="Generations", unit="generation"):
            fitness_scores = [fitness_function(individual, features_graph, train_labels) for individual in population]
            # print the max fitness value
            print("\nMax fitness:", max(fitness_scores))
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
