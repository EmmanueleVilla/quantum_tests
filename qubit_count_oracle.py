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
from qiskit.circuit.library import QFT, GroverOperator, MCXGate
from qiskit import QuantumCircuit, transpile, assemble, Aer, execute
from qiskit.visualization import plot_bloch_multivector, plot_histogram
from qiskit.circuit.library.standard_gates import HGate
from qiskit.circuit.library import OR
import networkx as nx
from qiskit.visualization import circuit_drawer, plot_histogram
from qiskit_aer import AerSimulator

from dataset import create_dataset


# Diffusion from the Qiskit textbook

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


q_input = QuantumRegister(9, 'input')

qc = QuantumCircuit(q_input)
qc.h(range(9))

stat_prep = qc.to_instruction()
inv_stat_prep = qc.inverse().to_instruction()

graph = QuantumRegister(9, 'graph')  # graph inputs
counts = QuantumRegister(3, 'count')  # ancillas to count the number of rooms
oracle = QuantumRegister(1, 'oracle')  # final oracle toggle
anc = QuantumRegister(7, 'anc_diff') # ancilla for diffusion
c = ClassicalRegister(9, 'c') # measurements

qc = QuantumCircuit(graph, counts, oracle, anc, c)

qc.barrier(label="State preparation")
qc.append(stat_prep, graph)

qc.barrier(label="Oracle preparation")
qc.x(oracle[0])
qc.h(oracle[0])

num_iterations = 32

for i in range(num_iterations):
    qc.barrier(label="Oracle")
    # I want to count how many rooms there are in the 3 count qubits

    qc.h(counts)
    qc.barrier()
    for qubit in graph:
        qc.crz(np.pi / 4, qubit, counts[0])
        qc.crz(np.pi / 2, qubit, counts[1])
        qc.crz(np.pi, qubit, counts[2])
        qc.barrier()

    # append inverse Quantum Fourier Transform
    qft = QFT(3, do_swaps=True, inverse=True)
    qc.append(qft.to_instruction(), counts)

    mcx = MCXGate(3, "count", "010")

    qc.append(mcx, qargs=[counts[0], counts[1], counts[2], oracle[0]])

    qc.append(qft.inverse().to_instruction(), counts)

    qc.barrier()

    # Then I reset the oracle
    for qubit in graph:
        qc.crz(-np.pi, qubit, counts[2])
        qc.crz(-np.pi / 2, qubit, counts[1])
        qc.crz(-np.pi / 4, qubit, counts[0])
        qc.barrier()

    qc.h(counts)
    qc.barrier()

    qc.barrier(label="Diffusion")

    # inv state preparation + x
    qc.append(stat_prep, range(9))
    qc.x(range(9))

    qc.barrier()

    # Multi-controlled Z
    cnz(qc, 8, graph[::-1], anc)

    qc.barrier()

    # x + state preparation
    qc.x(range(9))
    qc.append(inv_stat_prep, range(9))

# Measurement
qc.measure(graph, c)

print(qc.draw("text"))


qasm_sim = Aer.get_backend('qasm_simulator')
shots = 25000
qc_transpiled = transpile(qc, qasm_sim)
results = qasm_sim.run(qc_transpiled).result()
answer = results.get_counts()
# take the answer with more than 5 measurements
sub = {key: val for key, val in answer.items() if val > 5}

print("Prediction:", sub)