import pandas as pd
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from recursiveRouteChoice import ModelDataStruct, RecursiveLogitModelEstimation
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
ratio_matrix = np.zeros((num_nodes, num_nodes))
lanes_matrix = np.zeros((num_nodes, num_nodes))

# Function to parse lanes
def parse_lanes(lanes):
    try:
        return float(lanes)
    except (ValueError, TypeError):
        return np.nan  # Use NaN for missing or invalid lanes data

# Populate matrices
for _, edge in edges.iterrows():
    if edge['u'] in node_to_index and edge['v'] in node_to_index:
        from_node = node_to_index[edge['u']]
        to_node = node_to_index[edge['v']]
        adjacency_matrix[from_node, to_node] = 1
        length_matrix[from_node, to_node] = edge['length']
        
        # Process 'ratio' feature with a small value adjustment
        ratio = edge['ratio'] if 'ratio' in edge else 1  # Default to 1 if ratio is missing
        ratio_matrix[from_node, to_node] = ratio * 1e6  # Multiply by a factor to increase its range

        # Process 'lanes' feature
        lanes_matrix[from_node, to_node] = parse_lanes(edge['lanes'])

# Handle NaN values in the lanes_matrix by filling them with the mean number of lanes
lanes_mean = np.nanmean(lanes_matrix[lanes_matrix != 0])
lanes_matrix = np.nan_to_num(lanes_matrix, nan=lanes_mean)

# Convert matrices to sparse format
incidence_mat = sparse.csr_matrix(adjacency_matrix)
length_matrix = sparse.csr_matrix(length_matrix)
ratio_matrix = sparse.csr_matrix(ratio_matrix)
lanes_matrix = sparse.csr_matrix(lanes_matrix)

# Check connectivity
n_components, labels = connected_components(csgraph=incidence_mat, directed=True, return_labels=True)
print(f"Number of strongly connected components: {n_components}")

if n_components > 1:
    largest_component = np.argmax(np.bincount(labels))
    mask = labels == largest_component
    incidence_mat = incidence_mat[mask][:, mask]
    length_matrix = length_matrix[mask][:, mask]
    ratio_matrix = ratio_matrix[mask][:, mask]
    lanes_matrix = lanes_matrix[mask][:, mask]
    num_nodes = incidence_mat.shape[0]
    print(f"Using largest connected component with {num_nodes} nodes")

# Scaling the matrices
# **Scale the length matrix**
length_data = length_matrix.data
length_data_transformed = np.log1p(length_data)  # Log scaling
length_matrix = sparse.csr_matrix((length_data_transformed, length_matrix.indices, length_matrix.indptr), shape=length_matrix.shape)

# **Use the directly scaled ratio matrix** - No further transformation needed
# ratio_matrix has already been scaled by multiplying by a factor.

# **Scale the ratio matrix**
ratio_data = ratio_matrix.data
ratio_data_transformed = np.log1p(ratio_data)  # Log scaling
ratio_matrix = sparse.csr_matrix((ratio_data_transformed, ratio_matrix.indices, ratio_matrix.indptr), shape=ratio_matrix.shape)


# **Scale the lanes matrix**
lanes_data = lanes_matrix.data
lanes_data_transformed = np.log1p(lanes_data)  # Log scaling
lanes_matrix = sparse.csr_matrix((lanes_data_transformed, lanes_matrix.indices, lanes_matrix.indptr), shape=lanes_matrix.shape)

# Combine all matrices into the data list
data_list = [length_matrix, ratio_matrix, lanes_matrix]
network_struct = ModelDataStruct(data_list, incidence_mat)

print("\nLength matrix stats:")
print("Min:", np.min(length_matrix.data))
print("Mean:", np.mean(length_matrix.data))
print("Max:", np.max(length_matrix.data))

print("\nRatio matrix stats:")
print("Min:", np.min(ratio_matrix.data))
print("Mean:", np.mean(ratio_matrix.data))
print("Max:", np.max(ratio_matrix.data))

print("\nLanes matrix stats:")
print("Min:", np.min(lanes_matrix.data))
print("Mean:", np.mean(lanes_matrix.data))
print("Max:", np.max(lanes_matrix.data))

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
print('obs_index_list qty', len(obs_index_list))

# Step 4: Grid search for a better initial beta
best_beta = None
best_log_likelihood = float('-inf')
optimiser = optimisers.ScipyOptimiser(method='l-bfgs-b', options={'maxiter': 1000, 'ftol': 1e-8})

# Generate grid search for three betas, one for each feature
for beta1_init in np.linspace(-1, 0, 10):  # Adjust the range if needed
    for beta2_init in np.linspace(-1, 0, 10):  # Adjust the range if needed
        for beta3_init in np.linspace(-1, 0, 10):  # Adjust the range if needed
            model_est = RecursiveLogitModelEstimation(network_struct, observations_record=obs_index_list,
                                                      initial_beta=[beta1_init, beta2_init, beta3_init], mu=1,
                                                      optimiser=optimiser)
            log_likelihood = model_est.get_log_likelihood()[0]
            print(f"Initial betas: [{beta1_init}, {beta2_init}, {beta3_init}], Log-likelihood: {log_likelihood}")
            
            if log_likelihood < best_log_likelihood:
                best_log_likelihood = log_likelihood
                best_beta = [beta1_init, beta2_init, beta3_init]

print(f"Best initial beta: {best_beta} with Log-likelihood: {best_log_likelihood}")


# best_beta = [-0.3, -0.11, -0.6]  # -0.5555555555555556, -0.11111111111111116, -0.66


# Step 5: Use the best beta found as the initial value for the optimization
model_est = RecursiveLogitModelEstimation(network_struct, observations_record=obs_index_list,
                                          initial_beta=best_beta, mu=1,
                                          optimiser=optimiser)

log_likelihood_before = model_est.get_log_likelihood()[0]
beta_est = model_est.solve_for_optimal_beta(verbose=True)
log_likelihood_after = model_est.get_log_likelihood()[0]

print(f"Log-likelihood before optimization: {log_likelihood_before}")
print(f"Log-likelihood after optimization: {log_likelihood_after}")
print(f"Estimated beta: {beta_est}")

print(f" beta: [{beta_est[0]:6.4f}, {beta_est[1]:6.4f}, {beta_est[2]:6.4f}]")

# Print some statistics about the matrices
print("\nAdjacency matrix shape:", adjacency_matrix.shape)
print("Number of non-zero elements in adjacency matrix:", np.count_nonzero(adjacency_matrix))
print("Length matrix shape:", length_matrix.shape)
print("Number of non-zero elements in length matrix:", length_matrix.nnz)
print("Ratio matrix shape:", ratio_matrix.shape)
print("Number of non-zero elements in ratio matrix:", ratio_matrix.nnz)
print("Lanes matrix shape:", lanes_matrix.shape)
print("Number of non-zero elements in lanes matrix:", lanes_matrix.nnz)
