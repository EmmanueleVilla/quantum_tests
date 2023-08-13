from collections import deque

from qiskit import QuantumCircuit, Aer
from qiskit.circuit import Gate
from qiskit.circuit.library import CXGate, ZGate, CZGate
from qiskit.quantum_info import Statevector
import numpy as np

class QuantumGate:
    def __init__(self, gate_name, target_qubit, control_qubits=[]):
        self.gate_name = gate_name
        self.target_qubit = target_qubit
        self.control_qubits = control_qubits

    def __str__(self):
        return f"{self.gate_name}({self.target_qubit}, {self.control_qubits}"


def state_vector_to_string(vector: Statevector) -> str:
    return "".join(["+" if x >= 0 else "-" for x in vector.data])


def apply_operations(operations: [QuantumGate]) -> str:
    circuit = QuantumCircuit(2)
    circuit.h(range(2))
    for operation in operations:
        if operation.gate_name == "Z" and len(operation.control_qubits) == 0:
            circuit.z(operation.target_qubit)
        if operation.gate_name == "Z" and len(operation.control_qubits) == 1:
            circuit.cz(operation.control_qubits[0], operation.target_qubit)
    simulator = Aer.get_backend('statevector_simulator')
    simulator_result = simulator.run(circuit).result()
    simulator_vector: Statevector = simulator_result.get_statevector()
    vector_node = state_vector_to_string(simulator_vector)
    return vector_node


def bfs(target_vector, operations_list):
    visited = set()
    start_vector = "++++"
    queue = deque([(start_vector, [])])
    visited.add(tuple(map(tuple, start_vector)))

    while queue:
        current_vector, current_operations = queue.popleft()

        if np.array_equal(current_vector, target_vector):
            return current_vector, current_operations

        for operation in operations_list:
            if operation == "z":
                for qubit1 in range(2):
                    new_operations = current_operations + [QuantumGate("Z", qubit1, [])]
                    test_node = apply_operations(new_operations)
                    if test_node not in visited:
                        queue.append((test_node, new_operations))
                        visited.add(test_node)
    return None


def main():
    operations = ["z"]
    result = bfs("+-+-", operations)
    if result is not None:
        print("result:", result[0])
        print("operations:", "\n".join([f"{x.gate_name}({x.target_qubit}, {x.control_qubits})" for x in result[1]]))
    else:
        print("no result")

if __name__ == "__main__":
    main()
