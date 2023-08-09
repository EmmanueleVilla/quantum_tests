import networkx as nx
import random
import numpy as np
import json
import matplotlib.pyplot as plt
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.algorithms.optimizers import COBYLA
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.utils import algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN
from sklearn.model_selection import train_test_split

from graph_utils import check_graph_validity


if __name__ == "__main__":
    color_map = {0: 'white', 1: 'blue'}


    def generate_grid_graph():
        graph = nx.grid_2d_graph(3, 4)
        for node in graph.nodes:
            graph.nodes[node]['label'] = random.choice([0, 1])
        return graph


    valid = []
    invalid = []

    print(f"valid: {len(valid)}, invalid: {len(invalid)}")

    no = 0
    while len(valid) < 250 or len(invalid) < 250:
        try:
            G = generate_grid_graph()
            node_colors = [color_map[G.nodes[node]['label']] for node in G.nodes()]

            subgraph_nodes = [node for node in G.nodes() if G.nodes[node]['label'] == 1]
            subgraph = G.subgraph(subgraph_nodes)

            start_node = next(iter(subgraph.nodes()), None)

            t = nx.bfs_tree(subgraph, source=start_node)

            n1 = len(subgraph_nodes)
            n2 = t.number_of_nodes()

            if n1 == n2:
                valid.append(G)
            else:
                invalid.append(G)

            print(f"valid: {len(valid)}, invalid: {len(invalid)}")
        except:
            no += 1

    valid_data = []

    for G in valid:
        arr = [G.nodes[node]['label'] for node in G.nodes]
        valid_data.append((arr, 1))

    invalid_data = []

    for G in invalid:
        arr = [G.nodes[node]['label'] for node in G.nodes]
        invalid_data.append((arr, -1))

    invalid_data = invalid_data[:250]

    all_samples = valid_data + invalid_data
    random.shuffle(all_samples)
    split_ratio = 0.8
    split_idx = int(len(all_samples) * split_ratio)

    train_data = all_samples[:split_idx]
    test_data = all_samples[split_idx:]

    print(train_data)

    train_features, train_labels = zip(*train_data)
    test_features, test_labels = zip(*test_data)

    train_features = np.array(train_features)
    train_labels = np.array(train_labels)
    test_features = np.array(test_features)
    test_labels = np.array(test_labels)

    print(train_features)
    print(train_labels)

    print(test_features)
    print(test_labels)


    # Circuito di convoluzione

    def conv_circuit(thetas):
        target = QuantumCircuit(2)
        target.rz(-np.pi / 2, 1)
        target.cx(1, 0)
        target.rz(thetas[0], 0)
        target.ry(thetas[1], 1)
        target.cx(0, 1)
        target.ry(thetas[2], 1)
        target.cx(1, 0)
        target.rz(np.pi / 2, 0)
        return target


    params = ParameterVector("θ", length=3)
    circuit = conv_circuit(params)
    print(circuit.draw(output='text'))


    # Layer di convoluzione

    def conv_layer(num_qubits, param_prefix):
        qc = QuantumCircuit(num_qubits, name="Conv Layer")
        qubits = list(range(num_qubits))
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            qc = qc.compose(conv_circuit(params[param_index: (param_index + 3)]), [q1, q2])
            qc.barrier()
            param_index += 3
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            qc = qc.compose(conv_circuit(params[param_index: (param_index + 3)]), [q1, q2])
            qc.barrier()
            param_index += 3

        qc_inst = qc.to_instruction()

        qc = QuantumCircuit(num_qubits)
        qc.append(qc_inst, qubits)
        return qc


    circuit = conv_layer(4, "θ")
    print(circuit.decompose().draw("text"))


    # Circuito di pooling

    def pool_circuit(params):
        target = QuantumCircuit(2)
        target.rz(-np.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)

        return target


    params = ParameterVector("θ", length=3)
    circuit = pool_circuit(params)
    print(circuit.draw("text"))


    def pool_layer(sources, sinks, param_prefix):
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="Pooling Layer")
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        for source, sink in zip(sources, sinks):
            qc = qc.compose(pool_circuit(params[param_index: (param_index + 3)]), [source, sink])
            qc.barrier()
            param_index += 3

        qc_inst = qc.to_instruction()

        qc = QuantumCircuit(num_qubits)
        qc.append(qc_inst, range(num_qubits))
        return qc


    sources = [0, 1]
    sinks = [2, 3]
    circuit = pool_layer(sources, sinks, "θ")
    print(circuit.decompose().draw("text"))

    ## MODELLO VERO
    # Creo il modello

    feature_map = ZFeatureMap(12, reps=3)

    print(feature_map.decompose().draw(output='text'))

    ansatz = QuantumCircuit(12, name="Ansatz")

    # Primo layer conv
    ansatz.compose(conv_layer(12, "с1"), list(range(12)), inplace=True)

    # Primo layer pool
    ansatz.compose(pool_layer([0, 1, 2, 3, 4], [5, 6, 7, 8, 9, 10, 11], "p1"), list(range(12)), inplace=True)

    # Secondo layer conv
    ansatz.compose(conv_layer(7, "c2"), list(range(5, 12)), inplace=True)

    # Secondo layer pool
    ansatz.compose(pool_layer([0, 1, 2], [3, 4, 5, 6], "p2"), list(range(5, 12)), inplace=True)

    # Terzo layer conv
    ansatz.compose(conv_layer(4, "c3"), list(range(8, 12)), inplace=True)

    # Terzo layer pool
    ansatz.compose(pool_layer([0, 1, 2], [3], "p3"), list(range(8, 12)), inplace=True)

    # Unisco featuremap e ansatz
    circuit = QuantumCircuit(12)
    circuit.compose(feature_map, range(12), inplace=True)
    circuit.compose(ansatz, range(12), inplace=True)

    # Creo l'osservabile
    observable = SparsePauliOp.from_list([("Z" + "I" * 11, 1)])

    # Creo l'estimator
    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
    )
    print(circuit.draw("text"))

    evals = []


    def callback_graph(weights, obj_func_eval):
        file1 = open("evals_3x4_quadris.txt", "a")  # append mode
        file1.write(f"{(obj_func_eval, weights)}\n")
        file1.close()
        print(f"{len(evals)}: {obj_func_eval}")
        evals.append(obj_func_eval)


    # Creo il classificatore
    classifier = NeuralNetworkClassifier(
        qnn,
        optimizer=COBYLA(maxiter=1000),
        callback=callback_graph,
        initial_point=np.asarray([1.28741081e+00, 3.51573064e+00, -4.80155676e-01, -3.39396399e-01,
                                  1.85965411e+00, 7.48509970e-04, 1.66130835e+00, 3.03291429e-02,
                                  2.38904697e-01, 5.43866734e-01, 1.87415485e+00, 1.81460104e+00,
                                  3.29783209e-01, 1.11243855e+00, 1.86824467e+00, 6.73653304e-01,
                                  5.47418730e-01, -1.14545344e-01, 1.93044417e-01, -1.02290811e+00,
                                  1.74167837e-01, 6.74296958e-01, 1.61960842e+00, 1.23694166e+00,
                                  1.10412223e+00, 1.03803731e+00, 2.06992896e+00, 5.87149014e-01,
                                  1.14116648e+00, 1.81624910e+00, -3.92089370e-01, 1.79238328e+00,
                                  1.42303619e+00, 2.48618684e-01, 9.23110902e-01, 9.41557982e-01,
                                  6.41299543e-03, 1.28772731e-01, 5.61506389e-01, 1.36714724e+00,
                                  6.28470705e-01, 3.15147441e-01, 2.36093473e-01, 1.36249334e-01,
                                  8.90250949e-01, 1.19196078e+00, 6.63822031e-01, 4.32612173e-01,
                                  1.56966925e+00, 4.49045695e-01, 7.17241428e-01, 9.71256374e-02,
                                  8.03815644e-01, 1.71479146e+00, 7.62787679e-01, 1.90425923e+00,
                                  -1.17603250e-01, 4.32038765e+00, 2.79437115e+00, 7.48819642e-02,
                                  7.46504306e-01, 6.32747300e-01, 1.39218809e+00, 1.48325230e+00,
                                  7.57599269e-01, -4.50581196e-01, -3.48385052e-01, 1.99797543e+00,
                                  1.59692387e-01, 3.10614031e-01, 1.64343949e+00, 1.56086015e+00,
                                  9.63787619e-01, 1.54620583e+00, 7.20127459e-01, -8.06464954e-02,
                                  -4.57364397e-01, 6.05469339e-01, 8.43634028e-01, 3.20527122e-01,
                                  1.39877559e+00, 6.91054184e-01, 2.24444194e-01, 7.82227482e-01,
                                  5.88740096e-02, 8.76364645e-01, 2.76665280e-01, 7.07493539e-01,
                                  8.17782727e-01, 6.64427289e-01, 2.09293672e+00, 3.53115821e-02,
                                  1.96089258e+00])
    )

    x = np.asarray(train_features)
    y = np.asarray(train_labels)

    classifier.fit(x, y)

    print(f"Accuracy from the train data : {np.round(100 * classifier.score(x, y), 2)}%")

    x = np.asarray(test_features)
    y = np.asarray(test_labels)
    print(f"Accuracy from the test data : {np.round(100 * classifier.score(x, y), 2)}%")
