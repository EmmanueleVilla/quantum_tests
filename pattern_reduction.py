import base64
import io
import json
from collections import deque

from matplotlib import pyplot as plt
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, Aer, execute
from qiskit.circuit import qpy_serialization
from qiskit.visualization import circuit_drawer


def get_neighbors(circ):
    neighbors = []
    for i in range(5 + 1):
        new_circ = circ.copy()
        new_circ.x(i)
        neighbors.append(new_circ)
    for i in range(5 + 1):
        for j in range(5 + 1):
            if i != j:
                new_circ = circ.copy()
                new_circ.cx(i, j)
                neighbors.append(new_circ)
    for i in range(5 + 1):
        for j in range(5 + 1):
            for k in range(5 + 1):
                if i != j and j != k and i != k:
                    new_circ = circ.copy()
                    new_circ.ccx(i, j, k)
                    neighbors.append(new_circ)
    return neighbors


def circ_to_key(circ):
    buf = io.BytesIO()
    qpy_serialization.dump(circ, buf)
    key = base64.b64encode(buf.getvalue()).decode('utf8')
    return key


def evaluate(circ):
    qc = circ.copy()
    qc.measure(range(5), range(5))
    qc.measure(5, 5)

    simulator = Aer.get_backend("qasm_simulator")
    job = execute(qc, simulator, shots=1000)
    result = job.result()

    return result.get_counts(qc).keys()


def check_equals(result, expected):
    result = sorted(result)
    expected = sorted(expected)
    return result == expected


def main():
    nodes = QuantumRegister(5, "node")
    arcs_ancilla = QuantumRegister(4, "arc")
    result_ancilla = QuantumRegister(2, "result")
    final_ancilla = QuantumRegister(1, "final")
    node_measurement = ClassicalRegister(5, "measure")
    result_measurement = ClassicalRegister(1, "result_measure")

    qc = QuantumCircuit(nodes, arcs_ancilla, result_ancilla, final_ancilla, node_measurement, result_measurement)

    qc.h(nodes)
    qc.barrier()

    qc.x(nodes[0])
    qc.cx(nodes[0], result_ancilla[0])
    qc.x(nodes[0])
    qc.barrier()

    qc.ccx(nodes[0], nodes[1], arcs_ancilla[0])
    qc.ccx(nodes[0], nodes[2], arcs_ancilla[1])
    qc.ccx(nodes[0], nodes[3], arcs_ancilla[2])
    qc.ccx(nodes[0], nodes[4], arcs_ancilla[3])

    qc.barrier()

    qc.x(arcs_ancilla[0])
    qc.x(arcs_ancilla[1])
    qc.x(arcs_ancilla[2])
    qc.x(arcs_ancilla[3])
    qc.mcx(arcs_ancilla, result_ancilla[1])

    qc.barrier()
    qc.x(result_ancilla[0])
    qc.x(result_ancilla[1])
    qc.ccx(result_ancilla[0], result_ancilla[1], final_ancilla[0])

    qc.measure(nodes, node_measurement)
    qc.measure(final_ancilla, result_measurement)

    simulator = Aer.get_backend("qasm_simulator")
    job = execute(qc, simulator, shots=1000)
    result = job.result()

    print(qc.draw("text"))
    expected = sorted(result.get_counts(qc).keys())

    nodes = QuantumRegister(5, "node")
    final_ancilla = QuantumRegister(1, "final")
    node_measurement = ClassicalRegister(5, "measure")
    result_measurement = ClassicalRegister(1, "result_measure")

    qc2 = QuantumCircuit(nodes, final_ancilla, node_measurement, result_measurement)
    qc2.h(range(5))
    qc2.barrier()
    qc2.x(range(1, 5))
    qc2.mcx([1, 2, 3, 4], 5)
    qc2.x(range(1, 5))
    qc2.barrier()
    qc2.x(5)
    qc2.barrier()
    qc2.x(0)
    qc2.cx(0, 5)
    qc2.x(0)
    qc2.barrier()
    qc2.measure(nodes, node_measurement)
    qc2.measure(final_ancilla, result_measurement)
    print(qc2.draw("text"))
    print(expected)
    job = execute(qc2, simulator, shots=1000)
    result = job.result()
    expected2 = sorted(result.get_counts(qc2).keys())
    print(expected2)
    print(expected == expected2)

    # Disegna il circuito utilizzando Matplotlib
    fig, ax = plt.subplots()
    circuit_drawer(qc2, output='mpl', ax=ax)
    ax.axis('on')  # Mantieni gli assi visibili

    plt.savefig('advanced_oracle.png', dpi=300)



"""
    base_qc = QuantumCircuit(nodes, final_ancilla, result_measurement, node_measurement)
    base_qc.h(nodes)

    visited = set()
    queue = deque([base_qc])
    visited.add(circ_to_key(base_qc))
    checked = 0
    while queue:
        current_qc = queue.popleft()

        result = evaluate(current_qc)
        checked += 1

        if checked % 100 == 0:
            print(checked)
            print(len(queue))
            print(len(visited))
            print(current_qc.draw("text"))
            print("--------")

        if check_equals(result, expected):
            file = open("pattern_reduction_result.txt", "a")
            file.write(current_qc.draw("text"))
            file.write("\n")
            file.close()

        neighbors = get_neighbors(current_qc)
        for neighbor in neighbors:
            if circ_to_key(neighbor) not in visited:
                queue.append(neighbor)
                visited.add(circ_to_key(neighbor))
"""

if __name__ == "__main__":
    main()
