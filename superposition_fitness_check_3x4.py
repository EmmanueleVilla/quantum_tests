import random

import numpy as np
from numpy import sort
from qiskit import QuantumCircuit, Aer, transpile

from build_circuit import conv_layer, pool_layer
from dataset import create_dataset


def create_ansatz(nqubits):
    qc = QuantumCircuit(12)

    # Primo layer conv
    qc.compose(conv_layer(4, "с1"), list(range(4)), inplace=True)

    # Primo layer pool
    qc.compose(pool_layer([0, 1], [2, 3], "p1"), list(range(4)), inplace=True)

    # Secondo layer conv
    qc.compose(conv_layer(2, "с2"), list(range(2, 4)), inplace=True)

    # Secondo layer pool
    qc.compose(pool_layer([0], [1], "p2"), list(range(2, 4)), inplace=True)

    return qc


def normalize_to_unit_length(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    normalized_vector = vector / norm
    return normalized_vector


def test_circuit_initialization(qc, data):
    meas = qc.copy()
    meas.measure(range(len(data[0]) + 1), range(len(data[0]) + 1))
    print(meas.draw("text"))

    simulator = Aer.get_backend('qasm_simulator')
    result = simulator.run(meas, shots=4096).result()
    counts = result.get_counts(meas)

    results = sorted([x[1:] for x in list(counts.keys())])
    print("Data:\t\t", sorted(data))
    print("Results:\t", results)

    print("Data length: ", len(data))
    print("Results length: ", len(results))
    print("Intersection length: ", len(set(data).intersection(results)))

    data_arr = np.asarray(sorted(data))
    results_arr = np.asarray(results)

    if not np.array_equal(data_arr, results_arr):
        print("Test failed! Retry...")
        return test_circuit_initialization(qc, data)

    print("Test passed!")


def state_vector_from_data(data):
    # I need to create a state vector with superposition of each sample + 0 and 1
    # The number of states is data.shape[1]
    data_full = []
    size = 2 ** data.shape[1]
    state_vector = np.asarray([0.0] * size)
    states = 0
    for state in data:
        full_state = "".join(str(x) for x in reversed(state))
        data_full.append(full_state)
        index = int(full_state, 2)
        state_vector[index] = 1
        states += 1
    state_vector = normalize_to_unit_length(state_vector)
    print("States: ", states)
    return state_vector, data_full


def create_circuit(data):
    # Creating a superposition of all the samples + 0 and 1
    state_vector, data_full = state_vector_from_data(data)

    # Now I have a state vector representing the superposition of all the samples + 0 and 1
    # Let's create it and check that the results are ok
    qc = QuantumCircuit(data.shape[1] + 1, data.shape[1] + 1)
    qc.initialize(state_vector, range(data.shape[1]))
    qc.barrier()

    # Test if the output of the circuit matches the data
    test_circuit_initialization(qc.copy(), data_full)

    return qc


def eval_circuit(qc):
    meas = qc.copy()
    meas.measure(range(5), range(5))
    simulator = Aer.get_backend('qasm_simulator')
    result = simulator.run(transpile(meas, simulator), shots=1024).result()
    counts = result.get_counts()
    results = [x[1:] for x in list(counts.keys())]
    return sorted([(x[::-1], x[0]) for x in results])


def eval_fitness(qc, individual, features_graph, train_labels):
    fitness = 0
    target = list(zip(features_graph, [str(x) for x in train_labels]))
    target.sort(key=lambda x: x[0])
    # print("Graphs:\t\t", target)
    meas = qc.copy()

    # print("base qc:\t", eval_circuit(meas))

    ansatz = create_ansatz(len(features_graph[0]))
    ansatz = ansatz.bind_parameters(individual)

    meas.compose(ansatz.copy(), range(4), inplace=True)

    # print("conv qc:\t", eval_circuit(meas))
    meas.cx(4, 3)

    meas.compose(ansatz.copy().inverse(), range(4), inplace=True)

    result = eval_circuit(meas)
    # print("oracle qc:\t", result)

    return len(intersection(result, target)) / len(target)


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def learn(qc, train_features, train_labels):
    ansatz = create_ansatz(len(train_features[0]))
    num_theta = ansatz.num_parameters

    print("num_theta", num_theta)

    individual = np.random.normal(0, np.pi / 10, num_theta)

    features_graph = [''.join(str(x) for x in row) for row in train_features]
    print(features_graph)

    fitness = eval_fitness(qc, individual, features_graph, train_labels)
    print(fitness)
    offset = 0.001
    while True:
        found = False
        for i in range(len(individual)):
            old = individual[i]
            individual[i] += np.random.uniform(-1 * offset, offset)
            new_fitness = eval_fitness(qc, individual, features_graph, train_labels)

            if new_fitness > fitness:
                fitness = new_fitness
                print("New fitness", fitness)
                i -= 1
                found = True
            else:
                individual[i] = old
        if found:
            offset = 0.001
        else:
            offset += 0.001
            if offset > 0.1:
                offset = 0.001


def main():
    # Create the data
    train_features, train_labels, test_features, test_labels = create_dataset(250, negative_value=0)

    print(train_features)
    print(train_labels)

    # Create the circuit
    qc = create_circuit(train_features)

    # now the circuit is ready to learn the data... if it works
    learn(qc, train_features, train_labels)


if __name__ == "__main__":
    main()
