import pandas as pd
import numpy as np
from scipy import sparse
from recursiveRouteChoice import ModelDataStruct

from recursiveRouteChoice.data_loading import load_tntp_node_formulation
from recursiveRouteChoice import RecursiveLogitModelPrediction, ModelDataStruct, \
    RecursiveLogitModelEstimation
from recursiveRouteChoice import optimisers

# Read the data
edges = pd.read_csv('edge.txt')
nodes = pd.read_csv('node.txt', sep=' ', header=None, names=['osmid', 'y', 'x'])
transit = pd.read_csv('transit.csv')

# Print information about nodes
print("Nodes columns:", nodes.columns)
print("First few rows of nodes:")
print(nodes.head())
print("\nNodes info:")
print(nodes.info())

# Print information about edges
print("\nEdges columns:", edges.columns)
print("First few rows of edges:")
print(edges.head())
print("\nEdges info:")
print(edges.info())

# Print information about transit
print("\nTransit columns:", transit.columns)
print("First few rows of transit:")
print(transit.head())
print("\nTransit info:")
print(transit.info())

# Convert osmid, u, and v to strings
nodes['osmid'] = nodes['osmid'].astype(str)
edges['u'] = edges['u'].astype(str)
edges['v'] = edges['v'].astype(str)

# Create a dictionary to map node osmid to a sequential index
node_to_index = {osmid: index for index, osmid in enumerate(nodes['osmid'])}

# Check if all edge nodes are in the node list
edge_nodes = set(edges['u']).union(set(edges['v']))
node_ids = set(nodes['osmid'])
missing_nodes = edge_nodes - node_ids
if missing_nodes:
    print(f"\nWarning: The following nodes are in the edge list but not in the node list: {missing_nodes}")

# Continue with the rest of your code...
num_nodes = len(nodes)
adjacency_matrix = np.zeros((num_nodes, num_nodes))
length_matrix = np.zeros((num_nodes, num_nodes))
lanes_matrix = np.zeros((num_nodes, num_nodes))

for _, edge in edges.iterrows():
    if edge['u'] in node_to_index and edge['v'] in node_to_index:
        from_node = node_to_index[edge['u']]
        to_node = node_to_index[edge['v']]
        adjacency_matrix[from_node, to_node] = 1
        length_matrix[from_node, to_node] = edge['length']
        
        # Handle 'lanes' column carefully
        if pd.notna(edge['lanes']):
            try:
                lanes = float(edge['lanes'])
            except ValueError:
                # If 'lanes' contains non-numeric values, default to 1
                lanes = 1
        else:
            lanes = 1
        lanes_matrix[from_node, to_node] = lanes

# Convert matrices to sparse format
incidence_mat = sparse.csr_matrix(adjacency_matrix)
length_matrix = sparse.csr_matrix(length_matrix)
lanes_matrix = sparse.csr_matrix(lanes_matrix)

data_list = [length_matrix, lanes_matrix]

network_struct = ModelDataStruct(data_list, incidence_mat)

# After loading data
length_scale = np.max(data_list[0])
capacity_scale = np.max(data_list[1])
data_list[0] = data_list[0] / length_scale
data_list[1] = data_list[1] / capacity_scale

print("Length range:", np.min(data_list[0]), np.max(data_list[0]))
print("Capacity range:", np.min(data_list[1]), np.max(data_list[1]))

# Initialize the model
# beta_sim = np.array([-0.08, -0.000015]) 
beta_sim = np.array([-0.8 * length_scale, -0.00015 * capacity_scale])
model = RecursiveLogitModelPrediction(network_struct, initial_beta=beta_sim, mu=0.1)

# Generate some observations
num_obs = 100
orig_indices = np.random.randint(0, num_nodes, num_obs)
dest_indices = np.random.randint(0, num_nodes, num_obs)
obs = model.generate_observations(origin_indices=orig_indices, dest_indices=dest_indices,
                                  num_obs_per_pair=1, iter_cap=2000, rng_seed=42)


# Estimate the model
optimiser = optimisers.ScipyOptimiser(method='l-bfgs-b')
beta_est_init = [-0.005, -0.25]  # Changed 0.25 to -0.25
model_est = RecursiveLogitModelEstimation(network_struct, observations_record=obs,
                                          initial_beta=beta_est_init, mu=1,
                                          optimiser=optimiser)

beta_est = model_est.solve_for_optimal_beta(verbose=False)

print(f"beta expected: [{beta_sim[0]:6.4f}, {beta_sim[1]:6.4f}],"
      f" beta_actual: [{beta_est[0]:6.4f}, {beta_est[1]:6.4f}]")

# Print some statistics about the matrices
print("\nAdjacency matrix shape:", adjacency_matrix.shape)
print("Number of non-zero elements in adjacency matrix:", np.count_nonzero(adjacency_matrix))
print("Length matrix shape:", length_matrix.shape)
print("Number of non-zero elements in length matrix:", length_matrix.nnz)
print("Lanes matrix shape:", lanes_matrix.shape)
print("Number of non-zero elements in lanes matrix:", lanes_matrix.nnz)