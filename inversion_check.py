import numpy as np
from qiskit import QuantumCircuit, Aer, transpile

from build_circuit import conv_layer, pool_layer
from superposition_fitness_check import normalize_to_unit_length


def base_circuit():
    qc = QuantumCircuit(4, 4)
    size = 2 ** 4
    state_vector = np.asarray([0.0] * size)
    data = ["1111", "1001", "1011", "1101"]
    for state in data:
        index = int(state, 2)
        state_vector[index] = 1
    state_vector = normalize_to_unit_length(state_vector)

    qc.initialize(state_vector)
    return qc

def main():

    # TEST BASE CIRCUIT
    qc = base_circuit()
    qc.measure(range(4), range(4))

    simulator = Aer.get_backend('qasm_simulator')
    job = simulator.run(qc, shots=512)
    result = job.result()
    counts = result.get_counts(qc)
    print(counts)

    # TEST BASE CIRCUIT + CONV
    qc = base_circuit()

    conv = conv_layer(4, "с1")
    individual = np.random.normal(0, np.pi / 10, conv.num_parameters)
    conv.assign_parameters(individual, inplace=True)
    qc.compose(conv, list(range(4)), inplace=True)
    qc.measure(range(4), range(4))

    simulator = Aer.get_backend('qasm_simulator')
    job = simulator.run(transpile(qc, simulator), shots=512)
    result = job.result()
    counts = result.get_counts(qc)
    print(counts)

    # TEST BASE CIRCUIT + CONV + INVERSION
    qc = base_circuit()

    conv = conv_layer(4, "с1")
    individual = np.random.normal(0, np.pi / 10, conv.num_parameters)
    conv.assign_parameters(individual, inplace=True)
    qc.compose(conv, list(range(4)), inplace=True)
    qc.compose(conv.inverse(), list(range(4)), inplace=True)
    qc.measure(range(4), range(4))

    simulator = Aer.get_backend('qasm_simulator')
    job = simulator.run(transpile(qc, simulator), shots=512)
    result = job.result()
    counts = result.get_counts(qc)
    print(counts)

    # TEST BASE CIRCUIT + CONV + POOL
    qc = base_circuit()

    conv = conv_layer(4, "с1")
    individual = np.random.normal(0, np.pi / 10, conv.num_parameters)
    conv.assign_parameters(individual, inplace=True)

    pool = pool_layer([0, 1], [2, 3], "p1")
    individual = np.random.normal(0, np.pi / 10, pool.num_parameters)
    pool.assign_parameters(individual, inplace=True)

    qc.compose(conv, list(range(4)), inplace=True)
    qc.compose(pool, list(range(4)), inplace=True)
    qc.measure(range(4), range(4))

    simulator = Aer.get_backend('qasm_simulator')
    job = simulator.run(transpile(qc, simulator), shots=512)
    result = job.result()
    counts = result.get_counts(qc)
    print(counts)

    # TEST BASE CIRCUIT + CONV + POOL + INVERSION
    qc = base_circuit()

    conv = conv_layer(4, "с1")
    individual = np.random.normal(0, np.pi / 10, conv.num_parameters)
    conv.assign_parameters(individual, inplace=True)

    pool = pool_layer([0, 1], [2, 3], "p1")
    individual = np.random.normal(0, np.pi / 10, pool.num_parameters)
    pool.assign_parameters(individual, inplace=True)

    qc.compose(conv, list(range(4)), inplace=True)
    qc.compose(pool, list(range(4)), inplace=True)
    qc.compose(pool.inverse(), list(range(4)), inplace=True)
    qc.compose(conv.inverse(), list(range(4)), inplace=True)

    qc.measure(range(4), range(4))

    simulator = Aer.get_backend('qasm_simulator')
    job = simulator.run(transpile(qc, simulator), shots=512)
    result = job.result()
    counts = result.get_counts(qc)
    print(counts)


if __name__ == "__main__":
    main()
