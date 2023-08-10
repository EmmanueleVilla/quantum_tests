import random

import numpy as np
from qiskit import QuantumCircuit, Aer, transpile
from sklearn.preprocessing import MinMaxScaler

from build_circuit import conv_layer, pool_layer

def create_ansatz():
    qc = QuantumCircuit(4)
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
    meas.measure(range(5), range(5))
    print(meas.draw("text"))

    simulator = Aer.get_backend('qasm_simulator')
    result = simulator.run(meas, shots=1024).result()
    counts = result.get_counts(meas)
    print(counts)

    # check that the keys of the counts are the same as the data_full
    for key in counts.keys():
        if key not in data:
            print("Test failed! Retry...")
            return test_circuit_initialization(qc, data)

    # and also the opposite
    for key in data:
        if key not in counts.keys():
            print("Test failed! Retry...")
            return test_circuit_initialization(qc, data)

    print("Test passed!")


def state_vector_from_data(data):
    # I need to create a state vector with superposition of each sample + 0 and 1
    # The number of states is 2^5
    data_full = []
    size = 2 ** 5
    state_vector = np.asarray([0.0] * size)
    for state in data:
        for i in range(2):
            full_state = "".join(str(x) for x in state) + str(i)
            data_full.append(full_state)
            index = int(full_state, 2)
            state_vector[index] = 1
    state_vector[0] = 1 / np.sqrt(len(data) * 2)
    state_vector = normalize_to_unit_length(state_vector)
    return state_vector, data_full


def create_circuit(data):

    # Creating a superposition of all the samples + 0 and 1
    state_vector, data_full = state_vector_from_data(data)

    # Now I have a state vector representing the superposition of all the samples + 0 and 1
    # Let's create it and check that the results are ok
    qc = QuantumCircuit(5, 5)
    qc.initialize(state_vector, range(5))

    # Test if the output of the circuit matches the data
    test_circuit_initialization(qc.copy(), data_full)

    return qc


def create_test_data():
    all_samples = [
        ([0, 0, 0, 0], 0),
        ([0, 0, 0, 1], 1),
        ([0, 0, 1, 0], 1),
        ([0, 0, 1, 1], 0),
        ([0, 1, 0, 0], 1),
        ([0, 1, 0, 1], 0),
        ([0, 1, 1, 0], 1),
        ([0, 1, 1, 1], 0),
        ([1, 0, 0, 0], 1),
        ([1, 0, 0, 1], 1),
        ([1, 0, 1, 0], 0),
        ([1, 0, 1, 1], 1),
        ([1, 1, 0, 0], 0),
        ([1, 1, 0, 1], 0),
        ([1, 1, 1, 0], 0),
        ([1, 1, 1, 1], 1)
    ]

    random.shuffle(all_samples)
    split_ratio = 0.7
    split_idx = int(len(all_samples) * split_ratio)

    train_data = all_samples[:split_idx]
    test_data = all_samples[split_idx:]

    train_features, train_labels = zip(*train_data)
    test_features, test_labels = zip(*test_data)

    train_features = np.array(train_features)
    train_labels = np.array(train_labels)
    test_features = np.array(test_features)
    test_labels = np.array(test_labels)

    return train_features, train_labels, test_features, test_labels


def eval_fitness(qc, individual, features_graph, train_labels):

    fitness = 0

    meas = qc.copy()
    ansatz = create_ansatz()
    ansatz = ansatz.bind_parameters(individual)

    meas.compose(ansatz.copy(), range(4), inplace=True)

    meas.cx(3, 4)

    inv_ansatz = create_ansatz()
    inv_ansatz = inv_ansatz.bind_parameters(individual * -1)
    meas.compose(inv_ansatz.copy(), range(4), inplace=True)

    #print(meas.decompose().draw("text"))
    meas.measure(range(5), range(5))

    simulator = Aer.get_backend('qasm_simulator')
    result = simulator.run(transpile(meas, simulator), shots=1024).result()
    counts = result.get_counts(meas)

    for count in counts:
        graph = count[:4]
        label = count[4]
        # find the corresponding feature index
        feature_idx = features_graph.index(graph)
        # check if the label is correct
        if label == train_labels[feature_idx]:
            fitness += counts[count]

    print(fitness)


def learn(qc, train_features, train_labels):

    ansatz = create_ansatz()
    num_theta = ansatz.num_parameters

    print("num_theta", num_theta)

    individual = np.random.normal(0, np.pi / 10, num_theta)

    features_graph = [''.join(str(x) for x in row) for row in train_features]
    print(features_graph)

    fitness = eval_fitness(qc, individual, features_graph, train_labels)


def main():
    # Create the data
    train_features, train_labels, test_features, test_labels = create_test_data()

    print(train_features)
    print(train_labels)

    # Create the circuit
    qc = create_circuit(train_features)

    # now the circuit is ready to learn the data... if it works
    learn(qc, train_features, train_labels)

if __name__ == "__main__":
    main()
