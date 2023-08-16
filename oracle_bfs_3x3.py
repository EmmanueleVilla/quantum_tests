import time
from collections import deque

from qiskit import QuantumCircuit, Aer, transpile, assemble
from qiskit.circuit import Gate
from qiskit.circuit.library import CXGate, ZGate, CZGate, CCZGate
from qiskit.quantum_info import Statevector
import numpy as np
from typing import List


class QuantumGate:
    def __init__(self, gate_name, target_qubit, control_qubits=[]):
        self.gate_name = gate_name
        self.target_qubit = target_qubit
        self.control_qubits = control_qubits

    def to_string(self):
        return f"{self.gate_name}({self.target_qubit}, {self.control_qubits})"

    def pretty(self):
        return f"{self.gate_name}({self.target_qubit}, {self.control_qubits})"


class QuantumNode:
    def __init__(self, vector: Statevector, operations: [QuantumGate], fitness: int):
        self.vector = vector
        self.operations = operations
        self.fitness = fitness
        self.repr = state_vector_to_string(vector)

    def __lt__(self, other):
        return self.fitness < other.fitness


def state_vector_to_string(vector: Statevector) -> str:
    return "".join(["+" if x >= 0 else "-" for x in vector.data])


def state_vector_to_coefficient(vector: Statevector) -> List[int]:
    return [1 if x >= 0 else -1 for x in vector.data]


def get_neighbors(target_vector: str, current_vector: Statevector, current_operations, next_operation) -> List[
    QuantumNode]:
    neighbors: [QuantumNode] = []
    print("-------------- Z --------------")
    if next_operation == "Z":
        for qubit1 in range(3 * 3):
            new_operations = current_operations + [QuantumGate("Z", qubit1, [])]
            test_node = current_vector.evolve(ZGate(), qargs=[qubit1])
            arr = f"{state_vector_to_coefficient(test_node)}"
            arr = arr.replace("[", "{").replace("]", "}")
            #print(f"list.Add(new QuantumGate(\"{new_operations[-1].pretty()}\", new[] {arr}));")
            fitness = sum(c1 == c2 for c1, c2 in zip(state_vector_to_coefficient(test_node), target_vector))
            neighbors.append(QuantumNode(test_node, new_operations, fitness))
    print("-------------- CZ --------------")
    if next_operation == "CZ":
        for qubit1 in range(3 * 3):
            for qubit2 in range(3 * 3):
                if qubit1 != qubit2:
                    new_operations = current_operations + [QuantumGate("CZ", qubit1, [qubit2])]
                    test_node = current_vector.evolve(CZGate(), qargs=[qubit1, qubit2])
                    arr = f"{state_vector_to_coefficient(test_node)}"
                    arr = arr.replace("[", "{").replace("]", "}")
                    #print(f"list.Add(new QuantumGate(\"{new_operations[-1].pretty()}\", new[] {arr}));")
                    fitness = sum(c1 == c2 for c1, c2 in zip(state_vector_to_string(test_node), target_vector))
                    neighbors.append(QuantumNode(test_node, new_operations, fitness))

    print("-------------- CCZ --------------")
    if next_operation == "CCZ":
        for qubit1 in range(3 * 3):
            for qubit2 in range(3 * 3):
                for qubit3 in range(3 * 3):
                    if qubit1 != qubit2 and qubit1 != qubit3 and qubit2 != qubit3:
                        new_operations = current_operations + [QuantumGate("CCZ", qubit1, [qubit2, qubit3])]
                        test_node = current_vector.evolve(CCZGate(), qargs=[qubit1, qubit2, qubit3])
                        arr = f"{state_vector_to_coefficient(test_node)}"
                        arr = arr.replace("[", "{").replace("]", "}")
                        #print(f"list.Add(new QuantumGate(\"{new_operations[-1].pretty()}\", new[] {arr}));")
                        fitness = sum(c1 == c2 for c1, c2 in zip(state_vector_to_string(test_node), target_vector))
                        neighbors.append(QuantumNode(test_node, new_operations, fitness))

    print("-------------- ICCZ --------------")
    if next_operation == "ICCZ":
        for qubit1 in range(3 * 3):
            for qubit2 in range(3 * 3):
                for qubit3 in range(3 * 3):
                    if qubit1 != qubit2 and qubit1 != qubit3 and qubit2 != qubit3:
                        new_operations = current_operations + [QuantumGate("ICCZ", qubit1, [qubit2, qubit3])]
                        test_node = current_vector.evolve(CCZGate(ctrl_state="00"), qargs=[qubit1, qubit2, qubit3])
                        arr = f"{state_vector_to_coefficient(test_node)}"
                        arr = arr.replace("[", "{").replace("]", "}")
                        print(f"list.Add(new QuantumGate(\"{new_operations[-1].pretty()}\", new[] {arr}));")
                        fitness = sum(c1 == c2 for c1, c2 in zip(state_vector_to_string(test_node), target_vector))
                        neighbors.append(QuantumNode(test_node, new_operations, fitness))
    if next_operation == "ICCZM":
        for qubit1 in range(3 * 3):
            for qubit2 in range(3 * 3):
                for qubit3 in range(3 * 3):
                    if qubit1 != qubit2 and qubit1 != qubit3 and qubit2 != qubit3:
                        new_operations = current_operations + [QuantumGate("ICCZM", qubit1, [qubit2, qubit3])]
                        test_node = current_vector.evolve(CCZGate(ctrl_state="01"), qargs=[qubit1, qubit2, qubit3])
                        arr = f"{state_vector_to_coefficient(test_node)}"
                        arr = arr.replace("[", "{").replace("]", "}")
                        print(f"list.Add(new QuantumGate(\"{new_operations[-1].pretty()}\", new[] {arr}));")
                        fitness = sum(c1 == c2 for c1, c2 in zip(state_vector_to_string(test_node), target_vector))
                        neighbors.append(QuantumNode(test_node, new_operations, fitness))

    return neighbors


def dfs(base_state_vector, target_vector, operations_list):
    visited: [str] = []
    best: QuantumNode = QuantumNode(base_state_vector, [], 0)
    nodes_to_explore: [QuantumNode] = [best]
    start = time.time()

    while nodes_to_explore:
        current: QuantumNode = nodes_to_explore.pop(0)
        # print(time.time() - start)
        # start = time.time()
        if current.repr in visited:
            continue

        if current.fitness > best.fitness:
            best = current

        visited.append(current.repr)

        if len(visited) % 50 == 0:
            print("-------------")
            print("visited:", len(visited))
            print("queued:", len(nodes_to_explore))
            print("best:", best.fitness)
            print("ops:", [x.to_string() for x in best.operations])
            file = open("oracle_dfs_3x3_night.txt", "w")
            file.write(f"best_fitness={best.fitness}\nops={[x.to_string() for x in best.operations]}\n")
            file.close()

        if np.array_equal(state_vector_to_string(current.vector), target_vector):
            return current

        neighbors: [QuantumNode] = []
        for operation in operations_list:
            neighbors += get_neighbors(target_vector, current.vector, current.operations, operation)

        break
        # print(len(neighbors))

        neighbors.sort(key=lambda x: x.fitness, reverse=True)  # Sort by fitness in descending order
        # print("neighbors:", len(neighbors))
        # Insertion sort to insert neighbors into the list while maintaining sorting order
        for neighbor in neighbors:
            if neighbor.repr in visited:
                continue

            if neighbor.repr in [x.repr for x in nodes_to_explore]:
                continue

            insertion_index = len(nodes_to_explore)
            i = 0
            for node in nodes_to_explore:
                if neighbor.fitness > node.fitness:
                    insertion_index = i
                    break
                i += 1
            nodes_to_explore.insert(insertion_index, neighbor)

    return best


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


def test_unitary():
    qc = QuantumCircuit(9)
    qc.h(0)
    qc.z(0)
    simulator = Aer.get_backend('unitary_simulator')
    result = simulator.run(transpile(qc, simulator)).result()
    unitary = result.get_unitary()
    res = np.ones((2 ** 9, 2 ** 9)) * np.asmatrix(unitary)
    print(res)


""""
def get_operations_list(qubits_count):
    test_unitary()

    unitaries = []

    circuit = QuantumCircuit(qubits_count)
    circuit.h(range(qubits_count))
    simulator = Aer.get_backend('unitary_simulator')
    result = simulator.run(transpile(circuit, simulator)).result()
    unitary = result.get_unitary()
    matrix = np.asmatrix(unitary)
    np.set_printoptions(precision=3)
    print(matrix)

    for qubit1 in range(qubits_count):
        qc = QuantumCircuit(qubits_count)
        qc.z(qubit1)
        simulator = Aer.get_backend('unitary_simulator')
        job = assemble(transpile(qc, simulator))
        result = simulator.run(job).result()
        unitary = result.get_unitary()
        unitaries.append(unitary)

    return unitaries

"""


def get_base_state_vector(num_qubits):
    qc = QuantumCircuit(num_qubits)
    qc.h(range(num_qubits))
    simulator = Aer.get_backend('statevector_simulator')
    simulator_result = simulator.run(transpile(qc)).result()
    simulator_vector: Statevector = simulator_result.get_statevector()
    return simulator_vector


def main():
    # get_operations_list(6)
    # return

    base_state_vector = get_base_state_vector(9)

    operations = ["Z", "CZ", "CCZ", "ICCZ", "ICCZM"]

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
    result = dfs(base_state_vector, target, operations)
    if result is not None:
        print("fitness is ", result.fitness)
        print("result:", result.vector)
        print("operations:\n",
              "\n".join([f"{x.gate_name}({x.target_qubit}, {x.control_qubits})" for x in result.operations]))
    else:
        print("no result")


if __name__ == "__main__":
    import cProfile

    main()

    # cProfile.run('main()')
