import networkx as nx
import numpy as np
from qiskit import QuantumCircuit, Aer, transpile, execute, QuantumRegister, ClassicalRegister

solution = [2.24399475, 2.11576648, 1.85930994, 1.73108167, 2.30810889, 1.60285339, 2.43633716, 1.92342407, 1.7951958, 2.82102197, 1.98753821, 2.6927937 ]

qc = QuantumCircuit(9)
qc.x(0)
qc.cry(solution[0], 0, 1)
qc.cry(solution[1], 0, 3)
qc.cry(solution[2], 3, 4)
qc.cry(solution[3], 3, 6)
qc.cry(solution[4], 1, 4)
qc.cry(solution[5], 1, 2)
qc.cry(solution[6], 6, 7)
qc.cry(solution[7], 4, 7)
qc.cry(solution[8], 4, 5)
qc.cry(solution[9], 2, 5)
qc.cry(solution[10], 7, 8)
qc.cry(solution[11], 5, 8)

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

print_results([x for x in counts.keys()])