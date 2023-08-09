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

### Results
A QCNN with a ZZFeatureMap, incorporating conv+pool layers, consisting of 63 parameters was trained for 100 cycles (45 minutes) on a 3x3 graph dataset. It achieved approximately 67% accuracy on the training data and 74% accuracy on the test data.

Another QCNN with a ZZFeatureMap, accompanied by conv+pool layers, containing 93 parameters, was trained for 1250 cycles (5 hours) on a 3x4 graph dataset. It reached an accuracy of [TRAIN_ACC] on the training data and [TEST_ACC] on the test data.
