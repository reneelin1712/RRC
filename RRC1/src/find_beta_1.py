import pandas as pd
import numpy as np
from scipy import sparse
from recursiveRouteChoice import ModelDataStruct, RecursiveLogitModelPrediction
import ast

# Read the data
edges = pd.read_csv('edge.txt')
nodes = pd.read_csv('node.txt', sep=' ', header=None, names=['osmid', 'y', 'x'])

# Print the first few rows of edges to understand the data structure
print("First few rows of edges:")
print(edges.head())

# Convert osmid, u, and v to strings
nodes['osmid'] = nodes['osmid'].astype(str)
edges['u'] = edges['u'].astype(str)
edges['v'] = edges['v'].astype(str)

# Create a dictionary to map node osmid to a sequential index
node_to_index = {osmid: index for index, osmid in enumerate(nodes['osmid'])}

# Create matrices
num_nodes = len(nodes)
adjacency_matrix = np.zeros((num_nodes, num_nodes))
length_matrix = np.zeros((num_nodes, num_nodes))
lanes_matrix = np.zeros((num_nodes, num_nodes))

def parse_lanes(lanes):
    if pd.isna(lanes):
        return 1
    try:
        return float(lanes)
    except ValueError:
        try:
            lanes_list = ast.literal_eval(lanes)
            return np.mean([float(lane) for lane in lanes_list])
        except:
            return 1

for _, edge in edges.iterrows():
    if edge['u'] in node_to_index and edge['v'] in node_to_index:
        from_node = node_to_index[edge['u']]
        to_node = node_to_index[edge['v']]
        adjacency_matrix[from_node, to_node] = 1
        length_matrix[from_node, to_node] = edge['length']
        lanes_matrix[from_node, to_node] = parse_lanes(edge['lanes'])

# Convert matrices to sparse format
incidence_mat = sparse.csr_matrix(adjacency_matrix)
length_matrix = sparse.csr_matrix(length_matrix)
lanes_matrix = sparse.csr_matrix(lanes_matrix)

# Scale the matrices
length_scale = np.max(length_matrix.data)
capacity_scale = np.max(lanes_matrix.data)
length_matrix = length_matrix / length_scale
lanes_matrix = lanes_matrix / capacity_scale

data_list = [length_matrix, lanes_matrix]
network_struct = ModelDataStruct(data_list, incidence_mat)

def get_matrix_range(matrix):
    if isinstance(matrix, (sparse.csr_matrix, sparse.csc_matrix, sparse.coo_matrix)):
        if matrix.nnz > 0:
            return matrix.data.min(), matrix.data.max()
        else:
            return 0, 0
    elif isinstance(matrix, sparse.dok_matrix):
        if len(matrix.keys()) > 0:
            values = list(matrix.values())
            return min(values), max(values)
        else:
            return 0, 0
    else:
        raise ValueError(f"Unsupported matrix type: {type(matrix)}")

def test_beta_values(network_struct, beta_range):
    for i in beta_range:
        beta_sim = np.array([-i, -i])  # Using the same value for both betas
        try:
            model = RecursiveLogitModelPrediction(network_struct, initial_beta=beta_sim, mu=1)
            exp_utility = model.get_exponential_utility_matrix()
            min_val, max_val = get_matrix_range(exp_utility)
            print(f"Beta: {beta_sim}, Exp utility range: {min_val:.6e}, {max_val:.6e}")
            if min_val > 1e-10 and max_val < 1e10:  # Arbitrary thresholds
                return beta_sim
        except ValueError as e:
            print(f"Error for beta {beta_sim}: {str(e)}")
    return None

# Test a range of beta values
beta_range = np.logspace(-5, 0, 30)  # Test from 1e-5 to 1
best_beta = test_beta_values(network_struct, beta_range)

if best_beta is None:
    print("Could not find suitable beta values. Please check your data.")
else:
    print(f"Best beta values found: {best_beta}")