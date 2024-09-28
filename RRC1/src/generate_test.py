import pandas as pd
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from recursiveRouteChoice import ModelDataStruct, RecursiveLogitModelEstimation,RecursiveLogitModelPrediction
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
index_to_node = {index: osmid for osmid, index in node_to_index.items()}  # Reverse mapping
print('index_to_node',index_to_node)


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

# # Generate grid search for three betas, one for each feature
# for beta1_init in np.linspace(-1, 0, 10):  # Adjust the range if needed
#     for beta2_init in np.linspace(-1, 0, 10):  # Adjust the range if needed
#         for beta3_init in np.linspace(-1, 0, 10):  # Adjust the range if needed
#             model_est = RecursiveLogitModelEstimation(network_struct, observations_record=obs_index_list,
#                                                       initial_beta=[beta1_init, beta2_init, beta3_init], mu=1,
#                                                       optimiser=optimiser)
#             log_likelihood = model_est.get_log_likelihood()[0]
#             print(f"Initial betas: [{beta1_init}, {beta2_init}, {beta3_init}], Log-likelihood: {log_likelihood}")
            
#             if log_likelihood > best_log_likelihood:
#                 best_log_likelihood = log_likelihood
#                 best_beta = [beta1_init, beta2_init, beta3_init]

# print(f"Best initial beta: {best_beta} with Log-likelihood: {best_log_likelihood}")


best_beta = [-1, -1, -1]  # Example initial values for the three attributes


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



# # Step 1: Load the test set
# test_data = pd.read_csv('test_CV0.csv')

# # Step 2: Convert the ori and des in the test set from edge numbers to node numbers
# # Get the original and destination node indices based on the edge n_id
# test_data['ori_node'] = test_data['ori'].apply(lambda x: node_to_index[edge_to_nodes[int(x)][0]])
# test_data['des_node'] = test_data['des'].apply(lambda x: node_to_index[edge_to_nodes[int(x)][1]])

# # Initialize the model with the best beta values for prediction
# model = RecursiveLogitModelPrediction(network_struct,
#                                       initial_beta=np.array(best_beta), mu=1)

# # # Use the converted ori_node and des_node as origin and destination pairs
# # orig_indices = test_data['ori_node'].values
# # dest_indices = test_data['des_node'].values

# # # Ensure that orig_indices and dest_indices are within the valid range
# # orig_indices = orig_indices.astype(int)
# # dest_indices = dest_indices.astype(int)

# # valid_indices = np.arange(num_nodes - 1)  # Exclude the last index reserved for the dummy state
# # orig_indices = np.clip(orig_indices, 0, num_nodes - 2)
# # dest_indices = np.clip(dest_indices, 0, num_nodes - 2)

# # print('orig_indices',orig_indices)
# # print('dest_indices',dest_indices)

# orig_indices = [133]
# dest_indices = [93]

# # Generate trajectories using the model
# obs = model.generate_observations(origin_indices=orig_indices, dest_indices=dest_indices,
#                                   num_obs_per_pair=1, iter_cap=2000, rng_seed=42)

# print(f"Generated {len(obs)} trajectories")

# # Convert node sequences back to edge sequences for saving
# def convert_nodes_to_edges(node_path, node_to_index, edges):
#     edge_list = []
#     for i in range(len(node_path) - 1):
#         u = node_path[i]
#         v = node_path[i + 1]
#         edge_id = edges[(edges['u'] == nodes.loc[u, 'osmid']) & (edges['v'] == nodes.loc[v, 'osmid'])]['n_id']
#         if not edge_id.empty:
#             edge_list.append(str(edge_id.values[0]))
#     return '_'.join(edge_list)

# generated_traj_df = pd.DataFrame({
#     'ori': test_data['ori'].values,
#     'des': test_data['des'].values,
#     'trajectory': ['_'.join(map(str, traj)) for traj in obs]
# })

# # Convert node paths back to edge paths
# generated_traj_df['trajectory_edges'] = generated_traj_df['trajectory'].apply(
#     lambda x: convert_nodes_to_edges(list(map(int, x.split('_'))), node_to_index, edges)
# )

# generated_traj_df.to_csv('generated_trajectories.csv', index=False)
# print("Generated trajectories saved to 'generated_trajectories.csv'")

#--------------------------------------------------
# # Step 1: Load the test set
# test_data = pd.read_csv('test_CV0.csv')

# # Step 2: Convert the ori and des from edge numbers to node numbers
# test_data['ori_node'] = test_data['ori'].apply(lambda x: node_to_index[edge_to_nodes[int(x)][0]])
# test_data['des_node'] = test_data['des'].apply(lambda x: node_to_index[edge_to_nodes[int(x)][1]])

# # Initialize the model with the best beta values for prediction
# model = RecursiveLogitModelPrediction(network_struct,
#                                       initial_beta=np.array(best_beta), mu=1)

# # Generate trajectories using the model for each pair of origin and destination
# generated_trajectories = []
# for index, row in test_data.iterrows():
#     ori_node = row['ori_node']
#     des_node = row['des_node']

#     print('ori_node', ori_node)
#     print('des_node', des_node)

#     try:
#         # Generate trajectory
#         obs = model.generate_observations(origin_indices=[ori_node], dest_indices=[des_node],
#                                           num_obs_per_pair=1, iter_cap=2000, rng_seed=42)
        
#         # Check if the observation is valid
#         if obs and len(obs[0]) > 0:
#             print('obs', obs)
#             generated_trajectories.append(obs[0][1:])
#         else:
#             print(f"No valid trajectory found for ori_node: {ori_node}, des_node: {des_node}")
#             generated_trajectories.append([])  # Append empty trajectory if none found

#     except Exception as e:
#         print(f"Error generating trajectory for ori_node: {ori_node}, des_node: {des_node}")
#         print(f"Exception: {e}")
#         generated_trajectories.append([])  # Append empty trajectory on error

# # Add generated trajectories (in node index format) to the DataFrame
# test_data['generated_trajectory_nodes'] = ['_'.join(map(str, traj)) if traj else '' for traj in generated_trajectories]

# # Save the DataFrame to CSV
# test_data.to_csv('trajectory_comparison.csv', index=False)
# print("Trajectory comparison saved to 'trajectory_comparison.csv'")

# # Function to convert a path from edge IDs to node IDs
# def convert_path_to_node_sequence(edge_path, edge_to_nodes):
#     edge_ids = list(map(int, edge_path.split('_')))
#     node_sequence = []

#     for edge_id in edge_ids:
#         u, v = edge_to_nodes[edge_id]
#         if not node_sequence:
#             node_sequence.append(u)  # Start with the 'u' of the first edge
#         node_sequence.append(v)  # Append the 'v' of each edge to complete the sequence

#     return '_'.join(node_sequence)

# # Add the new column 'path_nodes' to the test_data DataFrame
# test_data['path_nodes'] = test_data['path'].apply(lambda x: convert_path_to_node_sequence(x, edge_to_nodes))

# # Display the first few rows to verify the result
# print(test_data[['path', 'path_nodes']].head())

# # Save the updated DataFrame to a new CSV file
# test_data.to_csv('test_data_with_node_paths.csv', index=False)
# print("Updated test data saved to 'test_data_with_node_paths.csv'")

# # Function to convert osmids to node indices
# def convert_osmid_to_index(node_sequence, node_to_index):
#     osmids = node_sequence.split('_')
#     node_indices = [str(node_to_index[osmid]) for osmid in osmids]
#     return '_'.join(node_indices)

# # Apply the function to the path_nodes column to get the node indices
# test_data['path_node_indices'] = test_data['path_nodes'].apply(lambda x: convert_osmid_to_index(x, node_to_index))

# # Display the updated DataFrame to verify the results
# print(test_data[['path', 'path_nodes', 'path_node_indices']].head())

# # Save the updated DataFrame to a new CSV file
# test_data.to_csv('test_data_with_node_indices.csv', index=False)
# print("Updated test data saved to 'test_data_with_node_indices.csv'")