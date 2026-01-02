import numpy as np
import networkx as nx
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def GMPN(graph, hidden_dim, W_self, W_neigh, attr_name = 'h'):
    #graph is taken to be a nx graph object
    #hidden_dim is the dimension of all node embeddings a.k.a 'h'
    #W_self and W_neigh are the weight matrices

    # Aggregation Step
    msg_vecs = {}

    for node in graph.nodes:
        neighbors = list(graph.neighbors(node))

        if len(neighbors) == 0:
            msg_vecs[node] = np.zeros(hidden_dim)
            continue

        #neigh_feats -> neighbour features
        neigh_feats = np.stack([graph.nodes[n][attr_name] for n in neighbors])
        msg_vecs[node] = neigh_feats.mean(axis=0)

    # Updation Step
    h_new = {}

    #print(f'Old Embedding: {len(graph.nodes[0][attr_name])}')
    for node in graph.nodes:
        h_self = graph.nodes[node][attr_name]

        h_new[node] = sigmoid(
            h_self @ W_self + msg_vecs[node] @ W_neigh
        )

    # Update the memory 
    nx.set_node_attributes(
        graph,
        {node: h_new[node] for node in graph.nodes},
        name=attr_name
    )

    
    # print(f'New Embedding: {graph.nodes[0][attr_name].shape}')



if __name__ == '__main__':
    # graph = nx.karate_club_graph()
    # print(graph.nodes[10])

    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]

    G = to_networkx(data, node_attrs=['x', 'y'], to_undirected=True)

    # print(f"Nodes: {G.number_of_nodes()}")
    # print(f"Edges: {G.number_of_edges()}")
    # #print(G.nodes)
    
    hidden_dim = len(list(G.nodes[0]['x']))
    W_self = np.random.randn(hidden_dim, hidden_dim)
    W_neigh = np.random.randn(hidden_dim, hidden_dim)

    num_layers = 5
    
    for _ in range(num_layers):
        GMPN(G, hidden_dim, W_self, W_neigh, 'x')