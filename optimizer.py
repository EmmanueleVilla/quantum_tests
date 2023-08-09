import threading

import numpy as np
from qiskit import transpile, QuantumCircuit, Aer

from build_circuit import build_circuit
from dataset import create_dataset
import tkinter as tk


class OptimizationApp:
    def __init__(self, root):
        self.optimization_running = False

        self.root = root
        self.root.title("Optimization GUI")

        self.start_button = tk.Button(root, text="Start Optimization", command=self.start_optimization_thread)
        self.start_button.pack()

        self.stop_button = tk.Button(root, text="Stop Optimization", command=self.stop_optimization)
        self.stop_button.pack()

        self.root.geometry("800x600")

    # Modificare la funzione inizializzando la sovrapposizione di stati = a training set
    def fitness_function(self, theta, features, labels):
        backend = Aer.get_backend('qasm_simulator')

        accurate_predictions = 0

        qc = QuantumCircuit(13, 13)

        qnn = self.create_circuit()
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

    def optimize(self):

        num_theta = 93

        individual = np.random.uniform(0, np.pi / 4, num_theta)
        train_features, train_labels, test_features, test_labels = create_dataset(100)
        features_graph = [''.join(str(x) for x in row) for row in train_features]

        fitness = self.fitness_function(individual, features_graph, train_labels)
        failed_attempts = 0
        offset = 0.005

        while self.optimization_running:
            found = False
            for i in range(len(individual)):
                if not self.optimization_running:
                    return
                old = individual[i]
                individual[i] += np.random.uniform(-1 * offset, offset)
                new_fitness = self.fitness_function(individual, features_graph, train_labels)
                print(f"New fitness: {new_fitness}")
                if new_fitness > fitness:
                    fitness = new_fitness
                    self.save_individual(individual, "optimization_4x3.txt")
                    found = True
                else:
                    individual[i] = old

            if found:
                failed_attempts = 0
                offset = 0.005
            else:
                failed_attempts += 1

            if failed_attempts >= 10:
                offset *= 1.5
                offset = max(offset, 0.05)

    def start_optimization_thread(self):
        if not self.optimization_running:
            self.optimization_running = True
            threading.Thread(target=self.optimize).start()

    def stop_optimization(self):
        self.optimization_running = False
        print("STOPPP")

    @staticmethod
    def create_circuit():
        return build_circuit()

    @staticmethod
    def save_individual(genes, filename):
        with open(filename, "w") as f:
            f.write(" ".join(map(str, genes)))


if __name__ == "__main__":
    root = tk.Tk()
    app = OptimizationApp(root)
    root.mainloop()
