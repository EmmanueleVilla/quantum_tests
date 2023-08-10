import random

import networkx as nx
import numpy as np


def generate_grid_graph():
    graph = nx.grid_2d_graph(3, 4)
    for node in graph.nodes:
        graph.nodes[node]['label'] = random.choice([0, 1])
    return graph


def create_dataset(size, negative_value=-1):
    valid = []
    invalid = []

    no = 0
    while len(valid) < size or len(invalid) < size:
        try:
            new_graph = generate_grid_graph()

            subgraph_nodes = [node for node in new_graph.nodes() if new_graph.nodes[node]['label'] == 1]
            subgraph = new_graph.subgraph(subgraph_nodes)

            start_node = next(iter(subgraph.nodes()), None)

            t = nx.bfs_tree(subgraph, source=start_node)

            n1 = len(subgraph_nodes)
            n2 = t.number_of_nodes()

            if n1 == n2:
                valid.append(new_graph)

            if n1 != n2:
                invalid.append(new_graph)
        except:
            no += 1

    valid_data = []

    already_seen = []

    for new_graph in valid:
        arr = [new_graph.nodes[node]['label'] for node in new_graph.nodes]
        if arr in already_seen:
            continue
        valid_data.append((arr, 1))
        already_seen.append(arr)

    invalid_data = []

    for new_graph in invalid:
        arr = [new_graph.nodes[node]['label'] for node in new_graph.nodes]
        if arr in already_seen:
            continue
        invalid_data.append((arr, negative_value))
        already_seen.append(arr)

    invalid_data = invalid_data[:size]

    all_samples = valid_data + invalid_data
    random.shuffle(list(all_samples))
    split_ratio = 0.8
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
