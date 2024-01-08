import numpy as np
import pygad
from qiskit import QuantumCircuit, Aer, transpile, execute
from qiskit.circuit.library import MCMT, YGate, RYGate


def on_gen(ga_instance):
    print("Generation : ", ga_instance.generations_completed)
    print("Fitness of the best solution :", ga_instance.best_solution()[1])


def fitness_func(ga_instance, solution, solution_idx):
    qc = QuantumCircuit(9, 9)
    qc.x(0)
    qc.cry(solution[0], 0, 1)
    qc.cry(solution[1], 0, 3)

    qc.cry(solution[2], 3, 6)

    # gate from 3-1 to 4
    ccry_314_01 = RYGate(solution[3]).control(2, ctrl_state='01')
    qc.append(ccry_314_01, [3, 1, 4])

    ccry_314_10 = RYGate(solution[4]).control(2, ctrl_state='10')
    qc.append(ccry_314_10, [3, 1, 4])

    ccry_314_11 = RYGate(solution[5]).control(2, ctrl_state='11')
    qc.append(ccry_314_11, [3, 1, 4])

    qc.cry(solution[5], 1, 2)

    # gate from 6-4 to 7
    ccry_647_01 = RYGate(solution[6]).control(2, ctrl_state='01')
    qc.append(ccry_647_01, [6, 4, 7])

    ccry_647_10 = RYGate(solution[7]).control(2, ctrl_state='10')
    qc.append(ccry_647_10, [6, 4, 7])

    ccry_647_11 = RYGate(solution[8]).control(2, ctrl_state='11')
    qc.append(ccry_647_11, [6, 4, 7])

    # gate from 4-2 to 5
    ccry_425_01 = RYGate(solution[9]).control(2, ctrl_state='01')
    qc.append(ccry_425_01, [4, 2, 5])

    ccry_425_10 = RYGate(solution[10]).control(2, ctrl_state='10')
    qc.append(ccry_425_10, [4, 2, 5])

    ccry_425_11 = RYGate(solution[11]).control(2, ctrl_state='11')
    qc.append(ccry_425_11, [4, 2, 5])

    # gate from 7-5 to 8
    ccry_758_01 = RYGate(solution[12]).control(2, ctrl_state='01')
    qc.append(ccry_758_01, [7, 5, 8])

    ccry_758_10 = RYGate(solution[13]).control(2, ctrl_state='10')
    qc.append(ccry_758_10, [7, 5, 8])

    ccry_758_11 = RYGate(solution[14]).control(2, ctrl_state='11')
    qc.append(ccry_758_11, [7, 5, 8])


    qc.measure(range(9), range(9))

    qasm_sim = Aer.get_backend('qasm_simulator')
    transpiled = transpile(qc, qasm_sim)
    job = qasm_sim.run(transpiled, shots=10000)
    result = job.result()
    counts = result.get_counts()
    values = [val for val in counts.values()]

    fitness_high_priority = len(values)
    max = np.max(values)
    min = np.min(values)
    diff = max - min

    return fitness_high_priority * 10000 - diff + 1 / np.std(values)


num_variables = 15

variable_range = np.linspace(0, np.pi*2, dtype='float')

ga_instance = pygad.GA(num_generations=200,
                       num_parents_mating=4,
                       sol_per_pop=100,
                       num_genes=num_variables,
                       gene_type=np.float64,
                       gene_space=variable_range,
                       on_generation=on_gen,
                       fitness_func=fitness_func)

ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()

print("Best solution: {}".format(solution))
print("Fitness: {}".format(solution_fitness))

qc = QuantumCircuit(9, 9)
qc.x(0)
qc.cry(solution[0], 0, 1)
qc.cry(solution[1], 0, 3)

qc.cry(solution[2], 3, 6)

# gate from 3-1 to 4
ccry_314_01 = RYGate(solution[3]).control(2, ctrl_state='01')
qc.append(ccry_314_01, [3, 1, 4])

ccry_314_10 = RYGate(solution[4]).control(2, ctrl_state='10')
qc.append(ccry_314_10, [3, 1, 4])

ccry_314_11 = RYGate(solution[5]).control(2, ctrl_state='11')
qc.append(ccry_314_11, [3, 1, 4])

qc.cry(solution[5], 1, 2)

# gate from 6-4 to 7
ccry_647_01 = RYGate(solution[6]).control(2, ctrl_state='01')
qc.append(ccry_647_01, [6, 4, 7])

ccry_647_10 = RYGate(solution[7]).control(2, ctrl_state='10')
qc.append(ccry_647_10, [6, 4, 7])

ccry_647_11 = RYGate(solution[8]).control(2, ctrl_state='11')
qc.append(ccry_647_11, [6, 4, 7])

# gate from 4-2 to 5
ccry_425_01 = RYGate(solution[9]).control(2, ctrl_state='01')
qc.append(ccry_425_01, [4, 2, 5])

ccry_425_10 = RYGate(solution[10]).control(2, ctrl_state='10')
qc.append(ccry_425_10, [4, 2, 5])

ccry_425_11 = RYGate(solution[11]).control(2, ctrl_state='11')
qc.append(ccry_425_11, [4, 2, 5])

# gate from 7-5 to 8
ccry_758_01 = RYGate(solution[12]).control(2, ctrl_state='01')
qc.append(ccry_758_01, [7, 5, 8])

ccry_758_10 = RYGate(solution[13]).control(2, ctrl_state='10')
qc.append(ccry_758_10, [7, 5, 8])

ccry_758_11 = RYGate(solution[14]).control(2, ctrl_state='11')
qc.append(ccry_758_11, [7, 5, 8])

qc.measure(range(9), range(9))

qasm_sim = Aer.get_backend('qasm_simulator')
transpiled = transpile(qc, qasm_sim)
job = qasm_sim.run(transpiled, shots=10000)
result = job.result()
counts = result.get_counts()
values = [val for val in counts.values()]

print("number of paths: ", len(values))
print("min: ", min(values))
print("max: ", max(values))
print("deviation: ", np.std(values))
