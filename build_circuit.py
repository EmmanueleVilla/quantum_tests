import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


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


def pool_circuit(params):
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)

    return target


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


def build_circuit():
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

    return ansatz
