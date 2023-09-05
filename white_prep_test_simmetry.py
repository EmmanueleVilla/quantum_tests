import math

import networkx as nx
import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, Aer, execute, transpile
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


def print_results(results):
    return
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
    job = execute(circuit, sim, shots=50000)
    result = job.result()
    counts = result.get_counts(circuit)
    count_results(counts)
    print_results(counts.keys())
    counts = result.get_counts()
    # histogram = plot_histogram(counts, color='midnightblue')
    # histogram.savefig('histogram.png')


graph_qubits = QuantumRegister(9, 'graph')
first_measures = ClassicalRegister(9, 'first_measure')
second_measures = ClassicalRegister(9, 'second_measure')
third_measures = ClassicalRegister(9, 'third_measure')

qc = QuantumCircuit(graph_qubits, first_measures, second_measures, third_measures)
qc.measure(graph_qubits, first_measures)

run(qc)

qc = wn(qc, 9)
qc.measure(graph_qubits, second_measures)
run(qc)

G = nx.grid_2d_graph(3, 3)
mapping = {(i, j): i * 3 + j for i in range(3) for j in range(3)}
G = nx.relabel_nodes(G, mapping)

edges = list(G.edges())
undirected_edges = edges + [(v, u) for u, v in edges]

print(undirected_edges)

divider = 1.5
for i in range(9):
    hadamards = []
    done = []
    edges = [edge for edge in undirected_edges if edge[0] == i]
    while len(edges) > 0:
        current = edges.pop(0)
        done.append(current)
        if i != current[1] and current[1] not in hadamards:
            qc.cry(np.pi / divider, current[0], current[1]).c_if(first_measures[i], 0).c_if(second_measures[i], 1)
            divider += 0.05
            hadamards.append(current[1])
            additional_edges = [e for e in undirected_edges if e[0] == current[1]]
            for e in additional_edges:
                if e not in edges and e not in done:
                    edges.append(e)

image = circuit_drawer(qc, output='mpl')
image.savefig('circuit_image.png')

statevector_sim = Aer.get_backend('statevector_simulator')
job = execute(qc, statevector_sim)
result = job.result()
outputstate = result.get_statevector(qc, decimals=3)
print(outputstate)

count_non_zero = sum(1 for element in outputstate if element != 0)
print("Non zero elements: " + str(count_non_zero))
print("Tot values: " + str(len(outputstate)))

ones_state = [1 if element != 0 else 0 for element in outputstate]
print(ones_state)
count_non_zero = sum(1 for element in ones_state if element != 0)
print("Non zero elements: " + str(count_non_zero))

normalized_vector = np.asarray(ones_state) / np.linalg.norm(ones_state)
print(normalized_vector)

qc = QuantumCircuit(9)
qc.initialize(normalized_vector, range(9))
qc = transpile(qc,
               basis_gates=["u3", "u2", "u1", "cx", "id", "u0", "u", "p", "x", "y", "z", "h", "s", "sdg", "t", "tdg",
                            "rx", "ry", "rz", "sx", "sxdg", "cz", "cy", "swap", "ch", "ccx", "cswap", "crx", "cry",
                            "crz", "cu1", "cp", "cu3", "csx", "cu", "rxx", "rzz", "rccx", "rc3x", "c3x", "c3sqrtx",
                            "c4x"])
print(qc.draw("text"))


# symmetric state yeeeee


stat_prep = qc.to_instruction()
inv_stat_prep = qc.inverse().to_instruction()

qc = QuantumCircuit(10, 9)

qc.barrier(label="State preparation")
qc.append(stat_prep, range(9))

qc.barrier(label="Oracle preparation")
qc.x(9)
qc.h(9)

qc.barrier(label="Oracle")
# oracle
qc.ccx(7, 8, 9)

qc.barrier(label="Diffusion")

# state preparation + x
qc.append(stat_prep, range(9))
qc.x(range(9))

qc.barrier()

# Multi-controlled Z
qc.h(8)
qc.mct(list(range(8)), 8)
qc.h(8)

qc.barrier()

# x + state preparation
qc.x(range(9))
qc.append(inv_stat_prep, range(9))

qc.measure(range(9), range(9))

print("------------------")
print(qc.draw("text"))

sim = Aer.get_backend('qasm_simulator')
job = execute(qc, sim, shots=1000)
result = job.result()
counts = result.get_counts(qc)
count_results(counts)

