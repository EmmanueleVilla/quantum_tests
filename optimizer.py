import threading

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
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

        self.root.geometry("800x800")

        # Creazione dei grafici
        self.figure_theta = Figure(figsize=(5, 4), dpi=100)
        self.plot_theta = self.figure_theta.add_subplot(1, 1, 1)
        self.canvas_theta = FigureCanvasTkAgg(self.figure_theta, master=root)
        self.canvas_widget_theta = self.canvas_theta.get_tk_widget()
        self.canvas_widget_theta.pack()

        self.figure_fitness = Figure(figsize=(5, 4), dpi=100)
        self.plot_fitness = self.figure_fitness.add_subplot(1, 1, 1)
        self.canvas_fitness = FigureCanvasTkAgg(self.figure_fitness, master=root)
        self.canvas_widget_fitness = self.canvas_fitness.get_tk_widget()
        self.canvas_widget_fitness.pack()

    def update_theta_plot(self, theta_values):
        self.plot_theta.clear()
        self.plot_theta.scatter(range(len(theta_values)), theta_values, marker='o')
        self.plot_theta.set_title("Theta Values")
        self.plot_theta.set_xlabel("Theta Index")
        self.plot_theta.set_ylabel("Theta Value")
        self.canvas_theta.draw()

    def update_fitness_plot(self, fitness_values):
        self.plot_fitness.clear()
        self.plot_fitness.plot(fitness_values)
        self.plot_fitness.set_title("Fitness Values")
        self.plot_fitness.set_xlabel("Iteration")
        self.plot_fitness.set_ylabel("Fitness Value")
        self.canvas_fitness.draw()

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
        fitness_values = []
        while self.optimization_running:
            found = False
            for i in range(len(individual)):
                if not self.optimization_running:
                    return
                old = individual[i]
                individual[i] += np.random.uniform(-1 * offset, offset)

                new_fitness = self.fitness_function(individual, features_graph, train_labels)
                fitness_values.append(new_fitness)

                self.update_theta_plot(individual)
                self.update_fitness_plot(fitness_values)

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
