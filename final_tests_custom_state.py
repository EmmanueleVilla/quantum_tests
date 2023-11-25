import math

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit, Aer, execute, QuantumRegister, ClassicalRegister, transpile
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_histogram


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


def run(circuit):
    circuit.barrier()
    print(circuit.draw('text'))
    sim = Aer.get_backend('qasm_simulator')
    job = execute(circuit, sim, shots=50000)
    result = job.result()
    counts = result.get_counts(circuit)
    count_results(counts)
    print_results(counts.keys())
    print(len(counts))

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


def grover_diff(qc, nodes_qubits,edge_anc,ancilla,stat_prep,inv_stat_prep):
    qc.append(inv_stat_prep,qargs=nodes_qubits)
    qc.x(nodes_qubits)
    #====================================================
        #3 control qubits Z gate
    cnz(qc,len(nodes_qubits)-1,nodes_qubits[::-1],ancilla)
    #====================================================
    qc.x(nodes_qubits)
    qc.append(stat_prep,qargs=nodes_qubits)

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

normalized_vector = np.asarray(global_state_vector) / np.linalg.norm(global_state_vector)

print(normalized_vector)
qc = QuantumCircuit(9)
qc.initialize(normalized_vector, range(9))
qc = transpile(qc,
               basis_gates=["u3", "u2", "u1", "cx", "id", "u0", "u", "p", "x", "y", "z", "h", "s", "sdg", "t", "tdg",
                            "rx", "ry", "rz", "sx", "sxdg", "cz", "cy", "swap", "ch", "ccx", "cswap", "crx", "cry",
                            "crz", "cu1", "cp", "cu3", "csx", "cu", "rxx", "rzz", "rccx", "rc3x", "c3x", "c3sqrtx",
                            "c4x"])

from qiskit.visualization import circuit_drawer

image = circuit_drawer(qc.decompose().decompose().decompose().decompose(), output='mpl')
image.savefig('initialization.png')

exit(1)

# Definizione del numero di qubit
n_nodes = 9

stat_prep = qc.to_instruction()
inv_stat_prep = qc.inverse().to_instruction()

nodes_qubits = QuantumRegister(n_nodes, name='nodes')
edge_anc = QuantumRegister(2, name='edge_anc')
ancilla = QuantumRegister(n_nodes-2, name = 'cccx_diff_anc')
neg_base = QuantumRegister(1, name='check_qubits')
class_bits = ClassicalRegister(n_nodes, name='class_reg')
tri_flag = ClassicalRegister(3, name='tri_flag')
qc = QuantumCircuit(nodes_qubits, edge_anc, ancilla, neg_base, class_bits, tri_flag)



# Initialize quantum flag qubits in |-> state
qc.x(neg_base[0])
qc.h(neg_base[0])
# Initializing i/p qubits in superposition
qc.append(stat_prep,qargs=nodes_qubits)
qc.barrier()
# Calculate iteration count
iterations = math.floor(math.pi/4*math.sqrt(416/105))
# Calculate iteration count
for i in np.arange(iterations):
    qc.barrier()
    qc.cx(nodes_qubits[8], edge_anc[0])
    qc.cx(nodes_qubits[7], edge_anc[1])
    qc.ccx(edge_anc[0], edge_anc[1], ancilla[0])
    qc.barrier()
    grover_diff(qc, nodes_qubits,edge_anc,ancilla,stat_prep,inv_stat_prep)
qc.measure(nodes_qubits, class_bits)

run(qc)
