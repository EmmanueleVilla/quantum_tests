import time

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

from build_circuit import conv_layer, pool_layer
from dataset import create_dataset
from graph_utils import check_graph_validity


def create_ansatz(nqubits):
    qc = QuantumCircuit(nqubits)

    size = nqubits
    start = 0
    layer = 0
    index = 16
    while index > 1:
        # print("Layer: ", layer)
        # print("Conv layer with range: ", range(start, size))
        qc.compose(conv_layer(index, f"с{layer}"), range(start, size), inplace=True)
        mid = index // 2
        source = range(0, mid)
        sink = range(mid, index)
        # print("Pool layer with source: ", source, " and sink: ", sink)
        qc.compose(pool_layer(source, sink, f"p{layer}"), range(start, size), inplace=True)
        index = index // 2
        layer += 1
        diff = size - start
        start += diff // 2

    # Disegna il circuito utilizzando Matplotlib
    # fig, ax = plt.subplots()
    # circuit_drawer(qc, output='mpl', ax=ax)
    # ax.axis('on')  # Mantieni gli assi visibili

    # Mostra il grafico
    # plt.show()
    return qc

if __name__ == "__main__":

    train_features, train_labels, test_features, test_labels = create_dataset(350, negative_value=-1, m=4, n=4)

    train_features = np.array(train_features)
    train_labels = np.array(train_labels)
    test_features = np.array(test_features)
    test_labels = np.array(test_labels)

    # Creo il modello

    feature_map = ZFeatureMap(16, reps=3)

    ansatz = create_ansatz(16)

    # Unisco featuremap e ansatz
    circuit = QuantumCircuit(16)
    circuit.compose(feature_map, range(16), inplace=True)
    circuit.compose(ansatz, range(16), inplace=True)

    # Creo l'osservabile
    observable = SparsePauliOp.from_list([("Z" + "I" * 15, 1)])

    # Creo l'estimator
    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
    )
    print(circuit.draw("text"))

    evals = []
    times = []
    def callback_graph(weights, obj_func_eval):
        file1 = open("evals_4x4_qcnn_bis.txt", "a")  # append mode
        file1.write(f"{(obj_func_eval, weights)}\n")
        file1.close()
        evals.append(obj_func_eval)
        times.append(int(time.time()))
        print(f"\n\ngen={len(times)}\nx={times}\ny={evals}\n\n")

    # Creo il classificatore
    classifier = NeuralNetworkClassifier(
        qnn,
        optimizer=COBYLA(maxiter=1, disp=True),
        callback=callback_graph,
        initial_point=np.asarray(
            [1.96195885, 1.20922309, 0.08466036, 2.06037602, 0.49982501,
             0.32996577, 2.22115656, 1.13094494, 0.86995768, 1.71845446,
             0.8042576, 0.52859457, 2.41241607, 1.39302354, 0.0902259,
             0.79637156, 2.58090943, 0.60727668, 1.4261875, 0.52005163,
             0.99290639, 2.5297394, 1.19169708, 1.37476328, 0.05843608,
             1.78255615, 0.57485482, 0.75374847, 0.68645397, 1.86461254,
             1.61922169, 0.62549614, 2.07101304, 1.73268849, 0.72287569,
             0.99915895, 1.433259, 2.22574628, 1.3216772, 1.06498817,
             0.32758402, 2.45017568, 0.99718482, 1.17007958, 1.41668421,
             0.90080423, 0.52437758, 0.8921536, 1.94412186, 2.48650971,
             1.45563508, 1.08549377, 1.01785719, 1.48026368, 1.36017061,
             0.62168223, 0.75031962, 0.52590343, 1.64176581, 1.02945652,
             1.18911684, 1.43430743, 0.59534797, 1.70242406, 0.2045366,
             1.10551437, 0.8921078, 0.95907294, 1.15307688, 1.74573784,
             0.15823973, 1.21200028, 1.02692969, 1.78190823, 1.09829951,
             2.36335531, 0.20563057, 0.11953478, 0.24147857, 1.63329194,
             1.14605601, 1.26173346, 1.52767994, 1.12109038, 0.52507247,
             -0.10636187, 1.47016244, 0.64983595, 1.18891901, 1.14137096,
             0.15850845, 1.33995136, 1.15849528, 0.44605299, 0.28837516,
             0.60912772, 1.19714121, 1.49254965, 1.97601284, 0.03830761,
             1.43245968, 1.83379418, 2.51536419, 1.03767247, 0.15589922,
             0.00805668, 1.27758697, 0.79792031, 0.22199241, 2.46006213,
             1.40920528, 1.16072444, 1.04654025, 1.21375779, 0.89262983,
             0.22049288, 1.15467807, 1.62700862, 1.04046092, 0.60514555,
             1.17530694, 0.56176787, 1.08834724, 1.53378946, 0.05039438,
             0.23805386, 0.98531282, 1.7084795, 1.41452201, 0.31178519,
             -0.00896455, 1.41210569, 0.68753887, 0.29997288, 1.22081324]
        ),
    )

    x = np.asarray(train_features)
    y = np.asarray(train_labels)

    classifier.fit(x, y)

    print(f"Accuracy from the train data : {np.round(100 * classifier.score(x, y), 2)}%")

    x = np.asarray(test_features)
    y = np.asarray(test_labels)
    print(f"Accuracy from the test data : {np.round(100 * classifier.score(x, y), 2)}%")
