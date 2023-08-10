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

        # Creazione del frame per i grafici
        self.graph_frame = tk.Frame(root)
        self.graph_frame.pack()

        # Creazione dei grafici
        self.figure = Figure(figsize=(10, 4), dpi=100)

        self.plot_theta = self.figure.add_subplot(1, 2, 1)
        self.plot_fitness = self.figure.add_subplot(1, 2, 2)

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.graph_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack()

        # Creazione del frame per il testo
        self.text_frame = tk.Frame(root)
        self.text_frame.pack()

        # Creazione del widget di testo con testo bianco
        self.text_view = tk.Text(self.text_frame, height=10, width=100)
        self.text_view.pack()

    def update_text_view(self, failed_attempts, fitness, offset):
        self.text_view.delete("1.0", tk.END)
        self.text_view.insert(tk.END, f"Failed Attempts: {failed_attempts}\n")
        self.text_view.insert(tk.END, f"Fitness: {fitness}\n")
        self.text_view.insert(tk.END, f"Offset: {offset}\n")

    def update_graphs(self, theta_data, fitness_data, max_fitness_values):
        self.plot_theta.clear()
        self.plot_fitness.clear()

        self.plot_theta.scatter(range(len(theta_data)), theta_data, marker='o')
        self.plot_theta.set_title("Theta Values")

        self.plot_fitness.plot(range(len(fitness_data)), fitness_data)
        self.plot_fitness.plot(range(len(fitness_data)), max_fitness_values)
        self.plot_fitness.set_title("Fitness Trend")

        self.canvas.draw()


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

        individual = np.random.normal(0, np.pi / 10, num_theta)
        train_features, train_labels, test_features, test_labels = create_dataset(100)
        features_graph = [''.join(str(x) for x in row) for row in train_features]

        fitness = self.fitness_function(individual, features_graph, train_labels)
        failed_attempts = 0
        offset = 0.001
        fitness_values = []
        max_fitness_values = []
        while self.optimization_running:
            found = False
            for i in range(len(individual)):
                if not self.optimization_running:
                    return
                self.update_graphs(individual, fitness_values, max_fitness_values)
                self.update_text_view(failed_attempts, fitness, offset)

                old = individual[i]
                individual[i] += np.random.uniform(-1 * offset, offset)

                new_fitness = self.fitness_function(individual, features_graph, train_labels)
                fitness_values.append(new_fitness)

                if new_fitness > fitness:
                    fitness = new_fitness
                    self.save_individual(individual, "optimization_4x3.txt")

                    # Try again to edit the same theta since it was successful
                    i -= 1
                    found = True
                else:
                    individual[i] = old

                max_fitness_values.append(fitness)

            if found:
                failed_attempts = 0
                offset = 0.001
            else:
                offset += 0.001
                offset = min(offset, 0.05)

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
