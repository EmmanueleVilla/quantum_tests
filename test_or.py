import math

import networkx as nx
import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, Aer, execute, transpile
from qiskit.quantum_info import Statevector
from qiskit.visualization import circuit_drawer, plot_histogram
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, Aer, transpile, execute, assemble, QuantumRegister, ClassicalRegister
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


# Crea un circuito quantistico con 3 qubit di controllo e 1 qubit target
graph_register = QuantumRegister(9, "graph")
ancilla_register = QuantumRegister(9, "anc")
measure = ClassicalRegister(9, "meas")
circuit = QuantumCircuit(graph_register, ancilla_register, measure)

circuit.h(graph_register)
circuit.barrier()
circuit.append(OR(3), [graph_register[0], graph_register[4],  graph_register[2],                      ancilla_register[1]])
circuit.append(OR(2), [graph_register[1], graph_register[5],                                          ancilla_register[2]])
circuit.append(OR(3), [graph_register[0], graph_register[4],  graph_register[6],                      ancilla_register[3]])
circuit.append(OR(4), [graph_register[1], graph_register[3],  graph_register[5], graph_register[7],   ancilla_register[4]])
circuit.append(OR(3), [graph_register[2], graph_register[4],  graph_register[8],                      ancilla_register[5]])
circuit.append(OR(2), [graph_register[3], graph_register[7],                                          ancilla_register[6]])
circuit.append(OR(3), [graph_register[4], graph_register[6],  graph_register[8],                      ancilla_register[7]])
circuit.append(OR(2), [graph_register[5], graph_register[7],                                          ancilla_register[8]])

for i in range(1, 9):
  circuit.ch(ancilla_register[i],graph_register[i])

circuit.barrier()

circuit.measure(graph_register, measure)

# Visualizza il circuito
print(circuit)

# Simula ed esegui il circuito
simulator = Aer.get_backend('qasm_simulator')
compiled_circuit = transpile(circuit, simulator)
job = execute(compiled_circuit, simulator, shots=10000)
result = job.result()
counts = result.get_counts()
print(counts)
print(len(counts.keys()))
# Visualizza il risultato
plot_histogram(counts).show()