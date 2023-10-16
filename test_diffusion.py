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
from qiskit.circuit.library import QFT, GroverOperator
from qiskit import QuantumCircuit, transpile, assemble, Aer, execute
from qiskit.visualization import plot_bloch_multivector, plot_histogram
from qiskit.circuit.library.standard_gates import HGate
from qiskit.circuit.library import OR
import networkx as nx
from qiskit.visualization import circuit_drawer, plot_histogram
from qiskit_aer import AerSimulator

from dataset import create_dataset

def print_results(results):
    for res in results:
        print_graph(res)


def print_graph(g):
    last = g.split(' ')[0]
    g = np.reshape([char for char in last], (3, 3))
    print_dungeon(g)

def print_dungeon(dungeon):
    print("---------")
    for row in dungeon:
        for column in row:
            print("□ " if column[0] == '0' else "■ ", end="")
        print()


def count_results(results):
    res = {}
    for result in results:
        converted = result.split(' ')[0]
        if converted in res:
            res[converted] += results[result]
        else:
            res[converted] = results[result]

    res = dict(sorted(res.items(), key=lambda x: x[1]))
    print(res)
    histogram = plot_histogram(res, color='midnightblue')
    histogram.savefig('histogram.png')

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
    print("start_point", start_point, ": ", len(counts.keys()))

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


def normalize_to_unit_length(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    normalized_vector = vector / norm
    return normalized_vector


global_state_vector = normalize_to_unit_length(global_state_vector)
print(global_state_vector)

qc = QuantumCircuit(9)
qc.initialize(global_state_vector, range(9))
qc = transpile(qc,
               basis_gates=["u3", "u2", "u1", "cx", "id", "u0", "u", "p", "x", "y", "z", "h", "s", "sdg", "t", "tdg",
                            "rx", "ry", "rz", "sx", "sxdg", "cz", "cy", "swap", "ch", "ccx", "cswap", "crx", "cry",
                            "crz", "cu1", "cp", "cu3", "csx", "cu", "rxx", "rzz", "rccx", "rc3x", "c3x", "c3sqrtx",
                            "c4x"])
#print(qc.draw("text"))

def cnz(qc, num_control, node, anc):
    """Construct a multi-controlled Z gate

    Args:
    num_control :  number of control qubits of cnz gate
    node :             node qubits
    anc :               ancillaly qubits
    """
    if num_control > 2:
        qc.ccx(node[0], node[1], anc[0])
        for i in range(num_control - 2):
            qc.ccx(node[i + 2], anc[i], anc[i + 1])
        qc.cz(anc[num_control - 2], node[num_control])
        for i in range(num_control - 2)[::-1]:
            qc.ccx(node[i + 2], anc[i], anc[i + 1])
        qc.ccx(node[0], node[1], anc[0])
    if num_control == 2:
        qc.h(node[2])
        qc.ccx(node[0], node[1], node[2])
        qc.h(node[2])
    if num_control == 1:
        qc.cz(node[0], node[1])


# symmetric state yeeeee

qc = QuantumCircuit(9)
qc.h(range(9))

stat_prep = qc.to_instruction()
inv_stat_prep = qc.inverse().to_instruction()

oracle_circuit = QuantumCircuit(9)
oracle_circuit.x(0)

# Definisci l'operatore di Grover (operatore di diffusione)
grover_op = GroverOperator(oracle=oracle_circuit, state_preparation=qc)

graph = QuantumRegister(9, 'graph')
oracle = QuantumRegister(1, 'oracle')
anc = QuantumRegister(7, 'anc')
c = ClassicalRegister(9, 'c')

qc = QuantumCircuit(graph, oracle, anc, c)

qc.barrier(label="State preparation")
qc.append(stat_prep, graph)


qc.barrier(label="Oracle preparation")
qc.x(9)
qc.h(9)

for i in range(1):
    qc.barrier(label="Oracle")
    # oracle
    qc.cx(8, 9)

    qc.barrier(label="Diffusion")

    # state preparation + x
    qc.append(stat_prep, range(9))
    qc.x(range(9))

    qc.barrier()

    # Multi-controlled Z
    cnz(qc, 8, graph[::-1], anc)

    qc.barrier()

    # x + state preparation
    qc.x(range(9))
    qc.append(inv_stat_prep, range(9))
#qc.measure(range(9), range(9))

state_vector = execute(qc, Aer.get_backend('statevector_simulator')).result().get_statevector()

image = circuit_drawer(qc, output='mpl')
image.savefig('circuit_image_2.png')



simulator = AerSimulator()
compiled_circuit = transpile(grover_op, simulator)
job = execute(compiled_circuit, simulator)
result = job.result()
counts = result.get_counts()
count_results(counts)

print_results([x for x in counts.keys()])
print(len(counts.keys()))
