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


q_input = QuantumRegister(8, 'input')
q_count = QuantumRegister(3, 'count')
q_input_measure = ClassicalRegister(8, 'input_measure')
q_count_measure = ClassicalRegister(3, 'count_measure')

qc = QuantumCircuit(q_input, q_count, q_input_measure, q_count_measure)
qc.h(q_input)
qc.barrier()
qc.h(q_count)
qc.barrier()

for qubit in q_input:
    qc.crz(np.pi / 4, qubit, q_count[0])
    qc.crz(np.pi / 2, qubit, q_count[1])
    qc.crz(np.pi, qubit, q_count[2])
    qc.barrier()

qc.barrier()

qft = QFT(3, do_swaps=True, inverse=True)
qc.append(qft.to_instruction(), q_count)

qc.barrier()

qc.measure(q_input, q_input_measure)
qc.measure(q_count, q_count_measure)

print(qc.draw('text'))

qasm_sim = Aer.get_backend('qasm_simulator')
shots = 25000
qc_transpiled = transpile(qc, qasm_sim)
results = qasm_sim.run(qc_transpiled).result()
answer = results.get_counts()
# get keys in answer
keys = []
for key in answer.keys():
    split = key.split(' ')
    # counts the 1s in the split[1] string
    true_count = split[1].count('1')
    # transform the split[0] binary string in decimal
    decimal = int(split[0], 2)
    print(f'{split[1]} ({true_count}) -> {split[0]} ({decimal})')
