import networkx as nx


def check_graph_validity(graphs):
    n_valid = 0
    n_invalid = 0
    for graph in graphs:
        try:
            G = nx.grid_2d_graph(3, 3)
            i = 0
            for node in G.nodes:
                G.nodes[node]['label'] = graph[i]
                i += 1
            subgraph_nodes = [node for node in G.nodes() if G.nodes[node]['label'] == '1']
            subgraph = G.subgraph(subgraph_nodes)

            start_node = next(iter(subgraph.nodes()), None)

            t = nx.bfs_tree(subgraph, source=start_node)

            n1 = len(subgraph_nodes)
            n2 = t.number_of_nodes()

            if n1 == n2:
                n_valid += 1
            else:
                n_invalid += 1
        except:
            n_invalid += 1
            pass
    return n_valid, n_invalid
