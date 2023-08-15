## Quantum Procedural Content Generation Tests

### Description
Welcome to the Quantum Procedural Content Generation Tests repository, where we explore the intriguing realm of generating intricate room layouts for dungeons, empowered by quantum computation techniques.

### Problem Statement
The challenge of crafting engaging and diverse procedural content for virtual environments has garnered significant attention. This repository addresses the intricate task of generating room layouts for dungeons. By leveraging the principles of quantum computation, we seek innovative solutions to create dynamic and captivating virtual worlds.

### Motivations
The intersection of procedural content generation and quantum computation presents an exciting frontier of exploration. Quantum computing's capacity to process vast amounts of data and perform complex operations in parallel provides an enticing avenue for tackling the complexities of dungeon design. Despite the current hardware limitations, our efforts strive to unlock novel insights that could revolutionize the field of procedural content generation.

### DataSet
We utilize the networkx library to generate random grid graphs. The aim is to gather both valid and invalid graphs for testing purposes. We conduct tests on grid graphs with dimensions 3x3 and 3x4.

### Disclaimer
The Quantum Neural Network used for the tests is the basic QNN offered by Qiskit, and it's not in any way tailored exclusively for this problem

### Ideas

#### Idea #1
- Train a Quantum Convolutional Neural Network (QCNN) to distinguish between valid and invalid graphs.
- Employ the QCNN as an oracle for the Grover algorithm.
- **Issue**: The QCNN with ZZFeatureMap takes inputs as parameters of gates, limiting its chaining with a Grover circuit.
- **Solution 1**: Experiment with encoding the dataset in a superposition rather than a feature map
- **Solution 2**: Train the QCNN without a feature map with a genetic algorithm.

#### Idea #2 in case the QCNN can't be embedded in Grover
- Train a QCNN to differentiate between valid and invalid graphs.
- Apply user-defined constraints to a Grover Algorithm (e.g., stipulate that the top left and top right cells are valid rooms).
- Leverage the QCNN to filter the results, considering the imposed constraints.
- **Issue**: This requires more measurements with different circuits!

#### Idea #3
- Train a Quantum Variational Auto-encoder to generate the maps
- TODO

### Tests

#### Test 1
A basic QCNN with a ZZFeatureMap, incorporating conv+pool layers, consisting of 63 parameters was trained for 100 cycles (45 minutes) on a 3x3 graph dataset. It achieved 67% accuracy on the training data and 74% accuracy on the test data.

#### Test 2
Another basic QCNN with a ZZFeatureMap, accompanied by conv+pool layers, containing 93 parameters, was trained for 1250 cycles (5 hours) on a 3x4 graph dataset. It reached an accuracy of 80% on the training data and 72% on the test data.

#### Test 3
A QCNN oracle trained with genetic algorithms with no feature map.
Gens: 50, population: 25, tournament size: 4, mutation rate: 0.1, mutation type: offset 0.1 and reset.
Start values between 0 and pi.
Start accuracy: 57%.
End accuracy: 58% lol
(12 sec per generation)

#### Test 4
A QCNN oracle trained with genetic algorithms with no feature map.
Gens: 50, population: 50, tournament size: 8, mutation rate: 0.1, mutation type: offset 0.05.
Start values between 0 and pi/4.
Start accuracy: 59%.
End accuracy: 56%, but it was 64% at some point, I swear.
(25 sec per generation)

#### Test 5
A custom optimizer that changes the theta values using backtracking to search for a local optima... The exploration seems to work! But we need to find a better way to check the fitness

#### Test 6
Check if the conv and pool layer are really invertible... conv OK, pool OK! (inversion_check.py)

#### Test 7
Try a new method to calculate the fitness and evolve the QCNN... using the basis encoding so we can merge the QCNN with Grover! (superposition_fitness_check.py). The oracle encoding and creation works, now trying with a bigger graph..

#### Test 8
Trying the oracle training with a 4x4 graph. Train data composed by nearly 700 graphs, 50:50 distribution. Trainable params: 135.

First try: Starting fitness is 0.4342. After 30min it's 0.4630

Second try with bigger initial jumps: Starting fitness is 0.4162. After 30min it's 0.4306

Third try: I found out there's a COBYLA optimizer that can be customized. trying with that one. Starting fitness at 18:30 is 0.38. After 30min it's 0.43

Forth try with custom optimizer: Starting fitness is 0.39. After 8 hours its 0.42

It seems the model doesn't work with this encoding...

#### Test 10
Try to find the oracle with a BFS on an operation list! (oracle_bfs_2.py) - it works!

#### Test 11
Upgrade to a 2x2 matrix. It works! NAISU

#### Test 12
Upgrade to a 2x3 matrix. The search works but it takes a long time because of the state vector simulation.

#### Test 13
Changed from simulator to matrix multiplication. Still a bit slow, but it's better. After only 100 nodes visited with DFS I have an oracle with 56/64 accuracy

#### Test 14
We should try to implement it in c++ since matrix multiplication takes a veeery long time in python