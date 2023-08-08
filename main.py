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
    file1 = open("evals_3x4.txt", "a")  # append mode
    file1.write(f"{(obj_func_eval, weights)}\n")
    file1.close()
    print(f"{len(evals)}: {obj_func_eval}")
    evals.append(obj_func_eval)


# Creo il classificatore
classifier = NeuralNetworkClassifier(
    qnn,
    optimizer=COBYLA(maxiter=500),
    callback=callback_graph
)

x = np.asarray(train_features)
y = np.asarray(train_labels)

classifier.fit(x, y)

print(f"Accuracy from the train data : {np.round(100 * classifier.score(x, y), 2)}%")

x = np.asarray(test_features)
y = np.asarray(test_labels)
print(f"Accuracy from the test data : {np.round(100 * classifier.score(x, y), 2)}%")

raise SystemExit(0)

qc = QuantumCircuit(9)
qc.h(range(9))
qc.measure_all()
sim = Aer.get_backend('qasm_simulator')
job = execute(qc, sim, shots=1024)
result = job.result()
counts = result.get_counts(qc)
print(counts)

n_valid = 0
n_invalid = 0

# for each count, create a graph
for count in counts:
    try:
        graph = nx.grid_2d_graph(3, 3)
        i = 0
        for node in graph.nodes:
            graph.nodes[node]['label'] = count[i]
            i += 1
        subgraph_nodes = [node for node in graph.nodes() if graph.nodes[node]['label'] == '1']
        subgraph = graph.subgraph(subgraph_nodes)

        start_node = next(iter(subgraph.nodes()), None)

        t = nx.bfs_tree(subgraph, source=start_node)

        n1 = len(subgraph_nodes)
        n2 = t.number_of_nodes()

        if n1 == n2:
            n_valid += 1 * counts[count]
        else:
            n_invalid += 1 * counts[count]
    except:
        n_invalid += 1 * counts[count]
        pass

# Estraendo a caso, sono validi circa il 40% dei grafi
print(f"Valid: {n_valid}, Invalid: {n_invalid}, Ratio: {n_valid / (n_valid + n_invalid)}")

# Proviamo con Grover.......

params = [0.26658639, 1.39185682, 1.26441732, 1.65066331, 0.70184443,
          0.07121344, -0.4059305, 1.87630911, 2.2962326, -0.3130339,
          -0.12599121, 0.22675333, 0.11061481, 1.99928554, 1.50205393,
          -0.11817186, 1.239293, 1.8140666, 0.91402014, 0.69623735,
          0.80278143, 0.94603847, 0.7436966, 0.16258697, 0.71631893,
          1.63787661, 1.49303928, 0.54952453, 0.18476963, 0.08714062,
          0.20842298, 0.51395874, 0.77461636, 0.29376057, 0.11167332,
          2.11782522, 1.6827945, 1.94598825, -0.19381761, 1.92049492,
          2.0297329, 1.59693153, 1.5357099, 0.78745197, -0.05649326,
          0.98311745, 0.92986878, 0.20321267, -0.5438011, 0.42558309,
          -0.20681997, 0.17114879, 1.78505035, 1.5579237, 0.69552092,
          -0.19142373, 1.21084039, 0.65426548, 0.19087339, 0.78856782,
          1.58671295, 0.52986756, 0.71776785]

print(ansatz.num_parameters)
ansatz.assign_parameters(params, inplace=True)

# State preparation
qc = QuantumCircuit(11, 9)
qc.h(range(9))
qc.x(9)
qc.h(9)

# Oracle
qc = qc.compose(ansatz)
qc.cx(8, 9)
qc = qc.compose(ansatz.inverse())

# get state vector
sim = Aer.get_backend('statevector_simulator')
job = execute(qc, sim)
state: qiskit.quantum_info.Statevector = job.result().get_statevector(qc)

# check if the oracle marked something, meaning one or more amplitude is negative

# save the list of the outputs with negative value
negatives = []
for key, value in state.to_dict().items():
    if value < 0:
        negatives.append(key)

print(negatives)
print(check_graph_validity(negatives))

# Diffusion
qc.barrier()
qc.h(range(9))
qc.x(range(9))
qc.barrier()
qc.cx(9, 10)
qc.barrier()
qc.x(range(9))
qc.h(range(9))
qc.barrier()

qc.measure(range(9), range(9))

#print(qc.draw("text"))

sim = Aer.get_backend('qasm_simulator')
job = execute(qc, sim, shots=1024)
result = job.result()
counts = result.get_counts(qc)
#print(counts)

n_valid = 0
n_invalid = 0

# for each count, create a graph
for count in counts:
    try:
        graph = nx.grid_2d_graph(3, 3)
        i = 0
        for node in graph.nodes:
            graph.nodes[node]['label'] = count[i]
            i += 1
        subgraph_nodes = [node for node in graph.nodes() if graph.nodes[node]['label'] == '1']
        subgraph = graph.subgraph(subgraph_nodes)

        start_node = next(iter(subgraph.nodes()), None)

        t = nx.bfs_tree(subgraph, source=start_node)

        n1 = len(subgraph_nodes)
        n2 = t.number_of_nodes()

        if n1 == n2:
            n_valid += 1 * counts[count]
        else:
            n_invalid += 1 * counts[count]
    except:
        n_invalid += 1 * counts[count]
        pass

#print(f"Valid: {n_valid}, Invalid: {n_invalid}, Ratio: {n_valid / (n_valid + n_invalid)}")
