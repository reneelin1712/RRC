import pandas as pd
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from recursiveRouteChoice import ModelDataStruct, RecursiveLogitModelPrediction, RecursiveLogitModelEstimation
from recursiveRouteChoice import optimisers
import ast

# Step 1: Read the data
edges = pd.read_csv('edge.txt')
nodes = pd.read_csv('node.txt', sep=' ', header=None, names=['osmid', 'y', 'x'])

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

# Populate matrices
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

# Check connectivity
n_components, labels = connected_components(csgraph=incidence_mat, directed=True, return_labels=True)
print(f"Number of strongly connected components: {n_components}")

if n_components > 1:
    largest_component = np.argmax(np.bincount(labels))
    mask = labels == largest_component
    incidence_mat = incidence_mat[mask][:, mask]
    length_matrix = length_matrix[mask][:, mask]
    lanes_matrix = lanes_matrix[mask][:, mask]
    num_nodes = incidence_mat.shape[0]
    print(f"Using largest connected component with {num_nodes} nodes")

# # Print detailed statistics about the matrices
# print("\nIncidence matrix:")
# print("Shape:", incidence_mat.shape)
# print("Non-zero elements:", incidence_mat.nnz)
# print("Density:", incidence_mat.nnz / (incidence_mat.shape[0] * incidence_mat.shape[1]))

# print("\nLength matrix:")
# print("Shape:", length_matrix.shape)
# print("Non-zero elements:", length_matrix.nnz)
# print("Min:", length_matrix.data.min())
# print("Max:", length_matrix.data.max())
# print("Mean:", length_matrix.data.mean())
# print("Median:", np.median(length_matrix.data))

# print("\nLanes matrix:")
# print("Shape:", lanes_matrix.shape)
# print("Non-zero elements:", lanes_matrix.nnz)
# print("Min:", lanes_matrix.data.min())
# print("Max:", lanes_matrix.data.max())
# print("Mean:", lanes_matrix.data.mean())
# print("Median:", np.median(lanes_matrix.data))

# Scale the matrices
length_scale = np.max(length_matrix.data)
capacity_scale = np.max(lanes_matrix.data)
length_matrix = length_matrix / length_scale
lanes_matrix = lanes_matrix / capacity_scale

data_list = [length_matrix, lanes_matrix]
network_struct = ModelDataStruct(data_list, incidence_mat)

print("\nAfter scaling:")
print("Length matrix stats:", np.min(length_matrix.data), np.mean(length_matrix.data), np.max(length_matrix.data))
print("Lanes matrix stats:", np.min(lanes_matrix.data), np.mean(lanes_matrix.data), np.max(lanes_matrix.data))

# Step 2: Create a mapping from edge n_id to node pairs (u, v)
edge_to_nodes = {}
for _, edge in edges.iterrows():
    edge_to_nodes[edge['n_id']] = (edge['u'], edge['v'])

# Function to convert a path from edge IDs to node IDs
def convert_path_to_nodes(path, edge_to_nodes):
    edge_ids = path.split('_')
    node_sequence = []

    for edge_id in edge_ids:
        u, v = edge_to_nodes[int(edge_id)]
        if not node_sequence:
            node_sequence.append(u)
        node_sequence.append(v)

    return node_sequence

# Convert node paths to index paths
def convert_path_to_indices(node_path, node_to_index):
    return [node_to_index[node] for node in node_path]

# Step 3: Convert paths in your observations
obs_data = pd.read_csv('path_edges.csv')
obs_data['node_path'] = obs_data['path'].apply(lambda x: convert_path_to_nodes(x, edge_to_nodes))
obs_data['index_path'] = obs_data['node_path'].apply(lambda x: convert_path_to_indices(x, node_to_index))

# Collect the list of observation paths in index form
obs_index_list = obs_data['index_path'].tolist()
print('obs_index_list qty',len(obs_index_list))

# Step 4: Estimate the model
# optimiser = optimisers.ScipyOptimiser(method='l-bfgs-b')
optimiser = optimisers.ScipyOptimiser(method='l-bfgs-b', options={'maxiter': 1000, 'ftol': 1e-8})
beta_est_init = [-0.04, -0.01]
model_est = RecursiveLogitModelEstimation(network_struct, observations_record=obs_index_list,
                                          initial_beta=beta_est_init, mu=0.5,
                                          optimiser=optimiser)

# beta_est = model_est.solve_for_optimal_beta(verbose=False)
log_likelihood_before = model_est.get_log_likelihood()[0]
beta_est = model_est.solve_for_optimal_beta(verbose=True)
log_likelihood_after = model_est.get_log_likelihood()[0]

print(f"Log-likelihood before optimization: {log_likelihood_before}")
print(f"Log-likelihood after optimization: {log_likelihood_after}")
print(f"Estimated beta: {beta_est}")

print(f" beta: [{beta_est[0]:6.4f}, {beta_est[1]:6.4f}]")

# Print some statistics about the matrices
print("\nAdjacency matrix shape:", adjacency_matrix.shape)
print("Number of non-zero elements in adjacency matrix:", np.count_nonzero(adjacency_matrix))
print("Length matrix shape:", length_matrix.shape)
print("Number of non-zero elements in length matrix:", length_matrix.nnz)
print("Lanes matrix shape:", lanes_matrix.shape)
print("Number of non-zero elements in lanes matrix:", lanes_matrix.nnz)
