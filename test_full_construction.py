import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, Aer, transpile, execute, assemble, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from qiskit.visualization.state_visualization import array_to_latex
from math import gcd
from numpy.random import randint
import pandas as pd
from fractions import Fraction
from qiskit.circuit.library import QFT
from qiskit import QuantumCircuit, transpile, assemble, Aer, execute
from qiskit.visualization import plot_bloch_multivector, plot_histogram
from qiskit.circuit.library.standard_gates import HGate
from qiskit.circuit.library import OR
import networkx as nx

from dataset import create_dataset

G = nx.grid_2d_graph(3, 3)
mapping = {(i, j): i * 3 + j for i in range(3) for j in range(3)}
G = nx.relabel_nodes(G, mapping)

edges = list(G.edges())
undirected_edges = edges + [(v, u) for u, v in edges]

print(undirected_edges)

global_state_vector = np.zeros(512)

for start_point in range(9):

    # Prepare the circuit
    graph = QuantumRegister(9)
    meas_one = ClassicalRegister(9)
    meas_two = ClassicalRegister(9)
    meas_three = ClassicalRegister(9)
    meas_four = ClassicalRegister(9)

    # Qubits measured until now
    measured = []
    measures = [meas_one, meas_two, meas_three, meas_four]
    measure_index = 0

    qc = QuantumCircuit(graph, meas_one, meas_two, meas_three, meas_four)

    # Set starting point to 1
    qc.x(start_point)
    qc.barrier()

    # Search the edges from start_point and put base h to them
    neighbors = [edge[1] for edge in undirected_edges if edge[0] == start_point]
    for neighbor in neighbors:
        qc.h(neighbor)

    qc.barrier()

    # First measurement and update measured qubits
    qc.measure([start_point] + neighbors, [meas_one[start_point]] + [meas_one[x] for x in neighbors])
    measured = measured + [start_point] + neighbors

    while len(neighbors) > 0:
        last_meas = measures[measure_index]
        measure_index += 1
        frontier = neighbors
        neighbors = list(set([edge[1] for edge in undirected_edges if edge[0] in frontier and edge[1] not in measured]))

        if len(neighbors) == 0:
            break

        # Foreach neighbor, check if it is already measured
        for neighbor in neighbors:
            # If it is not measured, check the incoming connections
            incoming = [edge[0] for edge in undirected_edges if edge[1] == neighbor and edge[0] in measured]
            if len(incoming) == 1:
                # If there is only one incoming connection, put an H if incoming is measured
                with qc.if_test((last_meas[incoming[0]], 1)):
                    qc.h(neighbor)
            if len(incoming) == 2:
                # If there are two incoming connections, put an H if at least one of them is measured
                with qc.if_test((last_meas[incoming[0]], 1)) as else_:
                    qc.h(neighbor)
                with else_:
                    with qc.if_test((last_meas[incoming[1]], 1)):
                        qc.h(neighbor)
            if len(incoming) == 3:
                # If there are three incoming connections, put an H if at least one of them are measured
                with qc.if_test((last_meas[incoming[0]], 1)) as else_:
                    qc.h(neighbor)
                with else_:
                    with qc.if_test((last_meas[incoming[1]], 1)) as else_:
                        qc.h(neighbor)
                    with else_:
                        with qc.if_test((last_meas[incoming[2]], 1)) as else_:
                            qc.h(neighbor)
            if len(incoming) == 4:
                # If there are four incoming connections, put an H if at least one of them are measured
                with qc.if_test((last_meas[incoming[0]], 1)) as else_:
                    qc.h(neighbor)
                with else_:
                    with qc.if_test((last_meas[incoming[1]], 1)) as else_:
                        qc.h(neighbor)
                    with else_:
                        with qc.if_test((last_meas[incoming[2]], 1)) as else_:
                            qc.h(neighbor)
                        with else_:
                            with qc.if_test((last_meas[incoming[3]], 1)) as else_:
                                qc.h(neighbor)

        qc.barrier()
        qc.measure(
            measured + neighbors,
            [measures[measure_index][x] for x in measured] + [measures[measure_index][x] for x in neighbors]
        )
        qc.barrier()
        measured = measured + neighbors

    # print(qc.draw("text"))
    simulator = Aer.get_backend('qasm_simulator')
    job = simulator.run(qc, shots=5000)
    result = job.result()
    counts = result.get_counts()
    # print(counts)
    # print(len(counts.keys()))

    statevectors = []



    for label, count in counts.items():
        for sub_label in label.split(" "):
            if sub_label != "0" * 9:
                statevectors.append(np.asarray(Statevector.from_label(sub_label)))

    # merge statevectors, keeping 1 if one of them is different from zero
    for i in range(512):
        for state_vector in statevectors:
            if state_vector[i] != 0:  # not np.isclose(0, state_vector[i], atol=1e-8, rtol=0):
                global_state_vector[i] = 1

    count_ones = np.sum(global_state_vector == 1)
    print(f"Numero di valori uguali a 1 in global_state_vectors: {count_ones}")

