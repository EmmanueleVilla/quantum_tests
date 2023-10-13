import math

import networkx as nx
import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, Aer, execute, transpile
from qiskit.quantum_info import Statevector
from qiskit.visualization import circuit_drawer, plot_histogram


def control_rotation(qc, control, target, theta):
    theta_dash = math.asin(math.cos(math.radians(theta / 2)))
    qc.u(theta_dash, 0, 0, target)
    qc.cx(control, target)
    qc.u(-theta_dash, 0, 0, target)
    return qc


def print_results(results):
    for res in results:
        print_graph(res)


def print_graph(g):
    last = g.split(' ')[0]
    g = np.reshape([char for char in last], (3, 3))
    print_dungeon(g)


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


# BEGIN
# Create the grid and save the edges

G = nx.grid_2d_graph(3, 3)
mapping = {(i, j): i * 3 + j for i in range(3) for j in range(3)}
G = nx.relabel_nodes(G, mapping)

edges = list(G.edges())
undirected_edges = edges + [(v, u) for u, v in edges]

print(undirected_edges)


# needed because when "returning back" the rotation, there are wrong states with negative amplitudes
def normalize_state_vector(state_vector):
    non_negative_data = [max(val, 0) for val in state_vector.data]

    norm = np.linalg.norm(non_negative_data)

    normalized_state_vector = Statevector(non_negative_data) / norm

    return normalized_state_vector


state_vectors = []
for i in range(9):
    # Start the simulation starting from each vertex of the graph
    qc = QuantumCircuit(9, 9)
    qc.x(i)
    visited = []
    queue = [i]
    counts = {}
    marked = [0]
    while len(queue) > 0:
        current = queue.pop(0)
        visited.append(current)
        neighbors = [edge[1] for edge in undirected_edges if edge[0] == current]
        for neighbor in neighbors:
            if neighbor not in marked:
                qc.ch(current, neighbor)
                marked.append(neighbor)
            if neighbor not in visited and neighbor not in queue:
                queue.append(neighbor)

        qc.barrier()

    state_vector_simulator = Aer.get_backend('statevector_simulator')
    job = execute(qc, state_vector_simulator)
    result = job.result()
    outputstate = result.get_statevector(qc, decimals=5)
    normalized = normalize_state_vector(outputstate)
    state_vectors.append(normalized)

    qc.barrier()
    qc.measure(range(9), range(9))
    run(qc)

    print(normalized)

global_state_vector = np.zeros(512)

for state_vector in state_vectors:
    print("Non zero values in state_vector ", np.count_nonzero(state_vector))

for i in range(512):
    for state_vector in state_vectors:
        if global_state_vector[i] == 0 and not np.isclose(0, state_vector[i], atol=1e-8, rtol=0):
            global_state_vector[i] = 1

count_ones = np.sum(global_state_vector == 1)
print(f"Numero di valori uguali a 1 in global_state_vectors: {count_ones}")

exit(1)

qc = wn(qc, 9)
qc.measure(graph_qubits, second_measures)
run(qc)

divider = 1.5
for i in range(1):
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
outputstate = result.get_statevector(qc, decimals=5)
print(outputstate)

count_non_zero = sum(1 for element in outputstate if not np.isclose(element, 0, atol=1e-8, rtol=0))
print("Non zero elements: " + str(count_non_zero))
print("Tot values: " + str(len(outputstate)))

ones_state = [1 if element != 0 else 0 for element in outputstate]
print(ones_state)
count_non_zero = sum(1 for element in ones_state if not np.isclose(element, 0, atol=1e-8, rtol=0))
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

stat_prep = qc.to_instruction()
inv_stat_prep = qc.inverse().to_instruction()

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
    qc.append(inv_stat_prep, range(9))
    qc.x(range(9))

    qc.barrier()

    # Multi-controlled Z
    cnz(qc, 8, graph[::-1], anc)

    qc.barrier()

    # x + state preparation
    qc.x(range(9))
    qc.append(stat_prep, range(9))

qc.measure(range(9), range(9))

print("------------------")
print(qc.draw("text"))

image = circuit_drawer(qc, output='mpl')
image.savefig('circuit_image_2.png')

sim = Aer.get_backend('qasm_simulator')
job = execute(qc, sim, shots=25000)
result = job.result()
counts = result.get_counts(qc)
count_results(counts)

print_results([x for x in counts.keys() if counts[x] > 500])

statevector_sim = Aer.get_backend('statevector_simulator')
job = execute(qc, statevector_sim)
result = job.result()
outputstate = result.get_statevector(qc, decimals=5)
print(outputstate)
