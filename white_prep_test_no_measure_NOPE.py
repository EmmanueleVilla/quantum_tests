import math

import networkx as nx
import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, Aer, execute
from qiskit.visualization import circuit_drawer, plot_histogram


def control_rotation(qc, control, target, theta):
    theta_dash = math.asin(math.cos(math.radians(theta / 2)))
    qc.u(theta_dash, 0, 0, target)
    qc.cx(control, target)
    qc.u(-theta_dash, 0, 0, target)
    return qc


def wn(circuit, n):
    for i in range(n):
        if i == 0:
            circuit.x(0)
            circuit.barrier()
        else:
            p = 1 / (n - (i - 1))
            theta = math.degrees(math.acos(math.sqrt(p)))
            theta = 2 * theta
            circuit = control_rotation(circuit, i - 1, i, theta)
            circuit.cx(i, i - 1)
            circuit.barrier()
    return circuit


def print_dungeon(dungeon):
    print("---------")
    for row in dungeon:
        for column in row:
            print("□ " if column[0] == '0' else "■ ", end="")
        print()


def print_results(results):
    for res in results:
        print_graph(res)


def print_graph(graph):
    last = graph.split(' ')[0]
    g = np.reshape([char for char in last], (3, 3))
    print_dungeon(g)


def run(circuit):
    circuit.barrier()
    print(circuit.draw('text'))
    sim = Aer.get_backend('qasm_simulator')
    job = execute(circuit, sim, shots=10000)
    result = job.result()
    counts = result.get_counts(circuit)
    print_results(counts.keys())
    counts = result.get_counts()
    histogram = plot_histogram(counts, color='midnightblue')
    histogram.savefig('histogram.png')


graph_qubits = QuantumRegister(9, 'graph')
first_measures = ClassicalRegister(9, 'first_measure')
second_measures = ClassicalRegister(9, 'second_measure')
third_measures = ClassicalRegister(9, 'third_measure')

qc = QuantumCircuit(graph_qubits, first_measures, second_measures, third_measures)

qc = wn(qc, 9)

G = nx.grid_2d_graph(3, 3)
mapping = {(i, j): i * 3 + j for i in range(3) for j in range(3)}
G = nx.relabel_nodes(G, mapping)

edges = list(G.edges())
undirected_edges = edges + [(v, u) for u, v in edges]

print(undirected_edges)

for i in range(9):
    hadamards = []
    done = []
    edges = [edge for edge in undirected_edges if edge[0] == i]
    while len(edges) > 0:
        current = edges.pop(0)
        done.append(current)
        if i != current[1] and current[1] not in hadamards:
            qc.ch(current[0], current[1])#.c_if(first_measures[i], 0).c_if(second_measures[i], 1)
            hadamards.append(current[1])
            additional_edges = [e for e in undirected_edges if e[0] == current[1]]
            for e in additional_edges:
                if e not in edges and e not in done:
                    edges.append(e)

image = circuit_drawer(qc, output='mpl')
image.savefig('circuit_image.png')

qc.measure(graph_qubits, third_measures)
run(qc)
