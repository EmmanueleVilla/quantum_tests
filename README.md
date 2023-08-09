## Quantum procedural content generation tests

### Description
Generating the room layout of a dungeon using quantum computation.

### DataSet
Using networkx, we generate random grid graphs until we manage to find N valid graphs and N invalid graphs.

Tests were done with 3x3 and 3x4 graphs until now.

### Ideas

#### Idea #1
- Train a Quantum Convolutional Neural Network to tell apart valid and invalid graphs
- Use the QCNN as oracle for the Grover algorithm
- PROBLEM: the QCNN with ZZFeatureMap takes the inputs as the parameters of the gates, so it can't be chained to a Grover circui
- SOLUTION: encode the dataset in a superposition instead of a featureMap may work

#### Idea #2
- Train a QCNN to telle apart valid and invalid graphs
- Apply user-defined constraint to a Grover Algorithm (e.g. I want that the top left cell and the top right cell are valid rooms)
- Use the QCNN to filter the constrained results
- PROBLEM: the QCNN must be trained without a feature map
- SOLUTION: train the QCNN using a genetic algorithm

### Results
A QCNN with a ZZFeatureMap and conv+pool layer with 63 parameters was trained for 100 cycles on a 3x3 graph dataset and reached around 67% accuracy on the train data and 74% accuracy on test data

A QCNN with a ZZFeatureMap and conv+pool layer with 93 parameters was trainer for 1250 cycles on a 3x4 graph dataset and reached around [TRAIN_ACC] accuracy on the train data and [TEST_ACC] on test data