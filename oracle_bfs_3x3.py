from collections import deque

from qiskit import QuantumCircuit, Aer, transpile
from qiskit.circuit import Gate
from qiskit.circuit.library import CXGate, ZGate, CZGate
from qiskit.quantum_info import Statevector
import numpy as np


class QuantumGate:
    def __init__(self, gate_name, target_qubit, control_qubits=[]):
        self.gate_name = gate_name
        self.target_qubit = target_qubit
        self.control_qubits = control_qubits

    def debug(self):
        return f"{self.gate_name}({self.target_qubit}, {self.control_qubits}"


def state_vector_to_string(vector: Statevector) -> str:
    return "".join(["+" if x >= 0 else "-" for x in vector.data])


def apply_operations(operations: [QuantumGate]) -> str:
    # print("***********************")
    # print("operations:", [x.debug() for x in operations])
    circuit = QuantumCircuit(9)
    circuit.h(range(9))
    for operation in operations:
        if operation.gate_name == "Z" and len(operation.control_qubits) == 0:
            circuit.z(operation.target_qubit)
        if operation.gate_name == "CZ" and len(operation.control_qubits) == 1:
            circuit.cz(operation.control_qubits[0], operation.target_qubit)
        if operation.gate_name == "CCZ" and len(operation.control_qubits) == 2:
            circuit.ccz(operation.control_qubits[0], operation.control_qubits[1], operation.target_qubit)

    # print(circuit.draw("text"))
    simulator = Aer.get_backend('statevector_simulator')
    simulator_result = simulator.run(transpile(circuit)).result()
    simulator_vector: Statevector = simulator_result.get_statevector()
    vector_node = state_vector_to_string(simulator_vector)
    return vector_node


def get_neighbor(current_operations, next_operation):
    if next_operation == "Z":
        for qubit1 in range(9):
            new_operations = current_operations + [QuantumGate("Z", qubit1, [])]
            test_node = apply_operations(new_operations)
            return test_node, new_operations
    if next_operation == "CZ":
        for qubit1 in range(9):
            for qubit2 in range(9):
                if qubit1 != qubit2:
                    new_operations = current_operations + [QuantumGate("CZ", qubit1, [qubit2])]
                    test_node = apply_operations(new_operations)
                    return test_node, new_operations


def bfs(target_vector, operations_list):
    visited = set()
    start_vector = ""
    queue = deque([(start_vector, [], 0)])
    visited.add(tuple(map(tuple, start_vector)))

    while queue:
        current_vector, current_operations, fitness = queue.popleft()

        if np.array_equal(current_vector, target_vector):
            return current_vector, current_operations

        for operation in operations_list:
            neighbor, new_operations = get_neighbor(current_operations, operation)
            if neighbor not in visited:
                fitness = sum(c1 == c2 for c1, c2 in zip(current_vector, target_vector))
                queue.append((neighbor, new_operations, fitness))
                visited.add(neighbor)
    return None


import networkx as nx


def check_graph_validity(graph):
    try:
        subgraph_nodes = [node for node in graph.nodes() if
                          graph.nodes[node]['label'] == 1 or graph.nodes[node]['label'] == '1']
        subgraph = graph.subgraph(subgraph_nodes)

        start_node = next(iter(subgraph.nodes()), None)

        t = nx.bfs_tree(subgraph, source=start_node)

        n1 = len(subgraph_nodes)
        n2 = t.number_of_nodes()

        return n1 == n2
    except:
        print("exception")
        return False


def main():
    operations = ["Z", "CZ"]

    target = ""
    for i in range(512):
        # i in binary
        binary = '{:09b}'.format(i)

        graph = nx.grid_2d_graph(3, 3)
        index = 0
        for node in graph.nodes:
            graph.nodes[node]['label'] = binary[index]
            index += 1

        valid = check_graph_validity(graph)
        if valid:
            target = target + "-"
        else:
            target = target + "+"
        # print(binary, " - is valid? ", valid)

    print("target is ", target)
    result = bfs(target, operations)
    if result is not None:
        print("result:", result[0])
        print("operations:\n",
              "\n".join([f"{x.gate_name}({x.target_qubit}, {x.control_qubits})" for x in result[1]]))
    else:
        print("no result")


if __name__ == "__main__":
    main()
