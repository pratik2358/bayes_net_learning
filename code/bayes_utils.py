import numpy as np
import matplotlib.pyplot as plt
from pomegranate.bayesian_network import _learn_structure, BayesianNetwork
import networkx as nx
from tqdm import tqdm
import pandas as pd
from scipy.stats import entropy
from collections import defaultdict, deque

variables_example = ['E', 'B', 'A', 'R', 'C']
dep_example = {0:(), 1:(), 2:(0,1), 3:(0,), 4:(2,)}
probs_example = {'E': [0.99, 0.01], 'B': [0.8, 0.2], 'A': [[0.86, 0.14], [0.03, 0.97], [0.1, 0.9], [0.01, 0.99]], 'R': [[0.95, 0.05], [0.4, 0.6]], 'C': [[0.99, 0.01], [0.3, 0.7]]}
data_sizes_example = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
n_example = 100

def convert_to_dict(dependency_tuple: tuple) -> dict:
    """
    input: tuple of dependencies
    -------------------------------------------
    output: dictionary of dependencies
    """
    return {i:dependency_tuple[i] for i in range(len(dependency_tuple))}

def convert_to_tuple(dependency_dict)->tuple:
    """
    input: dictionary of dependencies (order does not matter)
    -------------------------------------------
    output: tuple of dependencies
    """
    keys = list(dependency_dict.keys())
    keys.sort()
    tup =  [dependency_dict[key] for key in keys]
    return tuple(tup)

def visualize_BN(dependency_dict:dict = {0:(), 1:(), 2:()}, color = 'c', fig_name:str = 'BN.pdf') -> None:
    """
    input: dictionary of dependencies
    -------------------------------------------
    output: plot of the Bayesian Network
    """
    dependencies = convert_to_tuple(dependency_dict)
    G = nx.DiGraph()
    G.add_nodes_from(dependency_dict)
    for node, parents in dependency_dict.items():
      for p in parents:
        G.add_edge(p, node)

    independent_nodes = [i for i in G.nodes if G.in_degree(i) == 0 and G.out_degree(i) == 0]
    plt.figure(figsize=(6, 4))
    pos = nx.spring_layout(G, k=4, seed=9)
    nx.draw(G, pos, with_labels=True, node_color=color, node_size=2000, font_size=15, font_weight='bold', arrowsize=20)
    nx.draw_networkx_nodes(G, pos, nodelist=independent_nodes, node_color='orange', node_size=2000, alpha=0.6)
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20)
    plt.margins(0.5)
    # plt.title("Dependence Structure Graph")
    plt.savefig(fig_name)
    plt.show()

def topological_sort(dependency_dict)->list:
    """
    Topological sort of a directed acyclic graph (DAG) described by the dependency_dict
    input: dependency_dict
    -------------------------------------------
    output: topological order of the nodes
    """
    in_degree = defaultdict(int)
    for node in dependency_dict:
        if node not in in_degree:
            in_degree[node] = 0
        for parent in dependency_dict[node]:
            in_degree[node] += 1
            if parent not in in_degree:
                in_degree[parent] = 0

    # Collect nodes with no incoming edges
    zero_in_degree_queue = deque([node for node in in_degree if in_degree[node] == 0])
    
    topo_order = []

    while zero_in_degree_queue:
        node = zero_in_degree_queue.popleft()
        topo_order.append(node)

        # Reduce in-degree for all its neighbors
        for neighbor in dependency_dict:
            if node in dependency_dict[neighbor]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    zero_in_degree_queue.append(neighbor)

    # Check if there's a cycle (incomplete topological sorting)
    if len(topo_order) != len(in_degree):
        raise ValueError("The graph is not a DAG; it contains a cycle.")

    return topo_order

def simulate_data(variables: list, dependency: dict, probs: dict, data_size: int)->pd.DataFrame:
    """
    Simulate data based on the dependencies and probabilities
    input: variables, dependency, probs, data_size
    variables: list of variables
    dependency: dictionary of dependencies
    probs: dictionary of probabilities
    data_size: number of samples
    -------------------------------------------
    output: simulated data
    """
    ordering = topological_sort(dependency)
    X = np.zeros((data_size, len(variables)))
    X = pd.DataFrame(X, columns = variables)
    for key in ordering:
        if len(dependency[key]) == 0:
            X[variables[key]] = [1 if np.random.rand() < probs[variables[key]][1] else 0 for i in range(data_size)]
        elif len(dependency[key]) > 0:
            parents = dependency[key]
            for j in range(2**len(parents)):
                s = format(j, str(0)+str(len(parents))+"b")
                filter_condition = pd.Series([True] * len(X))
                for col, condition in zip([variables[pr] for pr in parents], [int(digit) for digit in s]):
                    filter_condition &= (X[col] == condition)
                X[variables[key]][filter_condition] = [1 if np.random.rand() < probs[variables[key]][j][1] else 0 for i in range(filter_condition.sum())]
    return X

def get_distribution(data:pd.DataFrame, variables:list, structure:tuple)->dict:
    """A function that returns the distribution of the variables in the Bayesian Network
    data: pandas DataFrame
    variables: list of variables in the Bayesian Network
    structure: tuple of parents of each variable in the Bayesian Network
    """
    dist = {x:None for x in variables}
    for i in range(len(structure)):
        if len(structure[i]) == 0:
            p = np.array([1 - data[variables[i]].mean(), data[variables[i]].mean()])
            p = p.reshape((2**len(structure[i]), 2))
            dist[variables[i]] = p
        else:
            parents = [variables[k] for k in structure[i]]
            p = np.zeros((2**len(parents), 2))
            for j in range(2**len(parents)):
                s = format(j, str(0)+str(len(parents))+"b")
                filter_condition = pd.Series([True] * len(data))
                for col, condition in zip(parents, [int(digit) for digit in s]):
                    filter_condition &= (data[col] == condition)
                if len(data[filter_condition])>0:
                    p[j, 0] = 1 - data[filter_condition][variables[i]].mean()
                    p[j, 1] = data[filter_condition][variables[i]].mean()
                else:
                    p[j, 0] = np.random.rand()
                    p[j, 1] = 1 - p[j, 0]
            p = p.reshape((2**len(parents), 2))
            dist[variables[i]] = p
    return dist

def avg_kl(dist1:dict, dist2:dict, eps:float = 1e-6)->float:
    """
    A function that calculates the average KL divergence between two distributions
    dist1: dictionary of distributions
    dist2: dictionary of distributions
    eps: small number to avoid division by zero

    """
    kl_div = 0
    dists = 0
    for key in dist1.keys():
        if dist1[key].shape[0] == dist2[key].shape[0]:    
            for i in range(dist1[key].shape[0]):
                kl_div += np.abs(entropy(dist1[key][i]+eps, dist2[key][i]+eps))
                dists += 1
    return kl_div/dists

def simulation_auto(variables:list = variables_example, dependency:dict = dep_example, probs:dict = probs_example, n:int = n_example, data_sizes:list = data_sizes_example, output:str = 'kl', algorithm:str = 'exact', max_parents:int = None)->np.ndarray:
    """
    A function to simulate the network learning experiment and return the average KL divergence or accuracy of the Bayesian Network structure learning
    variables: list of variables in the Bayesian Network
    dependency: dictionary of dependencies
    probs: dictionary of probabilities
    n: number of iterations
    data_sizes: list of data sizes
    output: 'kl' or 'accuracy'
    algorithm: 'exact' or 'chow-liu' used for learning the structure
    max_parents: maximum number of parents for each variable
    """
    if output == 'kl':
        kl_divs = np.zeros((len(data_sizes), n))
        for data_size in tqdm(data_sizes):
            for i in range(n):
                data = simulate_data(variables=variables, dependency=dependency, probs=probs, data_size=data_size)
                struct = _learn_structure(np.array(data, dtype = int), algorithm = algorithm, max_parents = max_parents)
                dist_data = get_distribution(data, variables, struct)

                model = BayesianNetwork(structure=struct)
                model.fit(np.array(data, dtype=int))
                dist_model = {x:None for x in variables}
                for k in range(len(variables)):
                    sh = list(model.distributions[k].parameters())[1].shape
                    dist_model[variables[k]] = list(model.distributions[k].parameters())[1].reshape(np.product(sh[:-1]), sh[-1])

                kl_divs[data_sizes.index(data_size), i] = avg_kl(dist_data, dist_model)
        return kl_divs
    elif output == 'accuracy':
        accs = np.zeros((len(data_sizes), n))
        for data_size in tqdm(data_sizes):
            for i in range(n):
                data = simulate_data(variables=variables, dependency=dependency, probs=probs, data_size=data_size)
                struct = _learn_structure(np.array(data, dtype = int), algorithm = algorithm, max_parents = max_parents)
                acc = np.mean([struct[i] == dependency[i] for i in range(len(struct))])
                accs[data_sizes.index(data_size), i] = acc
        return 100*accs
    
def simulate_joint_dist(data_sizes:list = data_sizes_example, variables:list = variables_example, probs:dict = probs_example, dependency:dict = dep_example, n:int = 100, eps:float = 1e-10, algorithm:str = 'exact', max_parents:int = None)->np.ndarray:
    """
    data_sizes: list of data sizes
    variables: list of variables in the Bayesian Network
    probs: dictionary of probabilities
    dependency: dictionary of dependencies
    n: number of iterations
    eps: small number to avoid division by zero
    algorithm: 'exact' or 'chow-liu' used for learning the structure
    max_parents: maximum number of parents for each variable
    """
    joint_dist_div = np.zeros((len(data_sizes), n))
    for d in tqdm(data_sizes):
        for i in range(n):
            data = simulate_data(variables = variables, dependency = dependency, probs = probs, data_size = d)
            prob_joint_data = {j:0 for j in range(2**len(variables))}
            for j in range(2**len(variables)):
                s = format(j, str(0)+str(len(variables))+"b")
                filter_condition = pd.Series([True] * len(data))
                for col, condition in zip([pr for pr in variables], [int(digit) for digit in s]):
                    filter_condition &= (data[col] == condition)
                prob_joint_data[j] = filter_condition.mean()

            struct = _learn_structure(np.array(data, dtype=int), algorithm = algorithm, max_parents = max_parents)
            model = BayesianNetwork(structure = struct)
            model.fit(np.array(data, dtype=int))

            data_model = pd.DataFrame(model.sample(n=d), columns = variables)
            prob_joint_model = {j:0 for j in range(2**len(variables))}
            for j in range(2**len(variables)):
                s = format(j, str(0)+str(len(variables))+"b")
                filter_condition = pd.Series([True] * len(data_model))
                for col, condition in zip([pr for pr in variables], [int(digit) for digit in s]):
                    filter_condition &= (data_model[col] == condition)
                prob_joint_model[j] = filter_condition.mean()
            joint_dist_div[data_sizes.index(d), i] = np.abs(entropy(np.array(list(prob_joint_data.values()))+eps, np.array(list(prob_joint_model.values()))+eps))
    return joint_dist_div

def marginals_from_data(data: pd.DataFrame)->dict:
    """
    A function that returns the average KL divergence between marginal probabilities of the variables in the original data and the learned Bayesian Network's sampled data
    data: pandas DataFrame
    -------------------------------------------
    output: average KL divergence
    """
    variables = list(data.columns)
    prob_joint_model = {j:np.zeros((2**(len(variables)-1),2)) for j in variables}
    for v in variables:
        for j in range(2**(len(variables)-1)):
            s = format(j, str(0)+str(len(variables)-1)+"b")
            filter_condition = pd.Series([True] * len(data))
            for col, condition in zip([pr for pr in list(set(variables)-{v})], [int(digit) for digit in s]):
                filter_condition &= (data[col] == condition)
            prob_joint_model[v][j][0] = 1 - filter_condition.mean()
            prob_joint_model[v][j][1] = 1 - prob_joint_model[v][j][0]
    return prob_joint_model

def simulation_marginals(variables:list = variables_example, dependency:dict = dep_example, probs:dict = probs_example, n:int = n_example, data_sizes:list = data_sizes_example, eps:float = 1e-10, algorithm:str = 'exact', max_parents:int = None)->np.ndarray:
    """
    A function that returns the average KL divergence between marginal probabilities of the variables in the original data and the learned Bayesian Network's sampled data
    variables: list of variables in the Bayesian Network
    dependency: dictionary of dependencies
    probs: dictionary of probabilities
    n: number of iterations
    data_sizes: list of data sizes
    eps: small number to avoid division by zero
    algorithm: 'exact' or 'chow-liu' used for learning the structure
    max_parents: maximum number of parents for each variable
    """
    kl_divs = np.zeros((len(data_sizes), n))
    for data_size in tqdm(data_sizes):
        for j in range(n):
            data = simulate_data(variables = variables, dependency = dependency, probs = probs, data_size = data_size)
            prob_joint = marginals_from_data(data)
            struct = _learn_structure(np.array(data, dtype = int), algorithm = algorithm, max_parents = max_parents)
            model = BayesianNetwork(structure = struct)
            model.fit(np.array(data, dtype = int))
            data_model = pd.DataFrame(model.sample(n = data_size), columns = variables)
            prob_joint_model = marginals_from_data(data_model)
            kl_divs[data_sizes.index(data_size), j] = np.abs(entropy(prob_joint['E']+eps, prob_joint_model['E']+eps, axis = 1).mean())
    return kl_divs