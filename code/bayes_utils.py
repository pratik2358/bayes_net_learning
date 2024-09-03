import numpy as np
import networkx as nx
from pomegranate.bayesian_network import _learn_structure
import matplotlib.pyplot as plt

def convert_to_tuple(dependency_dict: dict) -> tuple:
    """
    input: dictionary of dependencies (order does not matter)
    -------------------------------------------
    output: tuple of dependencies
    """
    keys = list(dependency_dict.keys())
    keys.sort()
    tup =  [dependency_dict[key] for key in keys]
    return tuple(tup)

def convert_to_dict(dependency_tuple: tuple) -> dict:
    """
    input: tuple of dependencies
    -------------------------------------------
    output: dictionary of dependencies
    """
    return {i:dependency_tuple[i] for i in range(len(dependency_tuple))}

def simulation(data_size = 10, noise1 = 0.2, noise2 = 0.3, n = 100, dependency_dict = dep, algorithm = 'exact'):
    """
    Simulate data and learn the structure of the Bayesian Network
    input: data_size, noise1, noise2, n, dependency_dict, algorithm
    data_size: number of samples to generate
    noise1: noise level for the first variable
    noise2: noise level for the second variable
    n: number of simulations
    dependency_dict: dictionary of dependencies
    algorithm: algorithm to learn the structure
    -------------------------------------------
    output: accuracy of the learned structure
    """
    exp_struct = convert_to_tuple(dependency_dict)
    correctness = []
    for i in range(n):
        v0 = np.random.randint(0,2, size = data_size)
        v1 = np.array([v0[i] if np.random.rand() > noise1 else 1-v0[i] for i in range(len(v0))])
        v2 = np.array([v0[i]*v1[i] if np.random.rand() > noise2 else 1-v0[i]*v1[i] for i in range(len(v0))])
        v3 = np.random.randint(0,2, size = data_size)
        X = np.hstack([v0.reshape(-1,1), v1.reshape(-1,1), v2.reshape(-1,1), v3.reshape(-1,1)])
        struct = _learn_structure(X, algorithm = algorithm)
        corr = 0
        for i in range(len(struct)):
            corr += int(struct[i] == exp_struct[i])
        corr /= len(struct)
        correctness.append(corr)
    return 100*np.mean(correctness)

def visualize_BN(dependency_dict:dict = {0:(), 1:(), 2:()}, fig_name:str = 'BN.pdf') -> None:
    """
    input: dictionary of dependencies
    -------------------------------------------
    output: plot of the Bayesian Network
    """
    dependencies = convert_to_tuple(dependency_dict)
    G = nx.DiGraph()
    G.add_nodes_from(range(len(dependencies)))
    for i, dep in enumerate(dependencies):
        for j in dep:
            G.add_edge(j, i)

    independent_nodes = [i for i in G.nodes if G.in_degree(i) == 0 and G.out_degree(i) == 0]
    plt.figure(figsize=(6, 4))
    pos = nx.spring_layout(G, k=1, seed=5)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=15, font_weight='bold', arrowsize=20)
    nx.draw_networkx_nodes(G, pos, nodelist=independent_nodes, node_color='green', node_size=2000, alpha=0.6)    
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20)
    plt.margins(0.5)
    plt.title("Dependence Structure Graph")
    plt.savefig(fig_name)
    plt.show()
