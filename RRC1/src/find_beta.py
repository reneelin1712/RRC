import pandas as pd
import numpy as np
from scipy import sparse
import scipy.sparse as sp
from recursiveRouteChoice import ModelDataStruct, RecursiveLogitModelPrediction, RecursiveLogitModelEstimation
from recursiveRouteChoice import optimisers

# Read the data
edges = pd.read_csv('edge.txt')
nodes = pd.read_csv('node.txt', sep=' ', header=None, names=['osmid', 'y', 'x'])
transit = pd.read_csv('transit.csv')

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

# Scale the matrices
length_scale = sp.csr_matrix.max(length_matrix)
lanes_scale = sp.csr_matrix.max(lanes_matrix)

if length_scale != 0:
    length_matrix_scaled = length_matrix / length_scale
else:
    length_matrix_scaled = length_matrix

if lanes_scale != 0:
    lanes_matrix_scaled = lanes_matrix / lanes_scale
else:
    lanes_matrix_scaled = lanes_matrix

# Add a small constant to avoid zero values
epsilon = 1e-6
length_matrix_scaled = length_matrix_scaled + epsilon * sparse.eye(length_matrix_scaled.shape[0], format='csr')
lanes_matrix_scaled = lanes_matrix_scaled + epsilon * sparse.eye(lanes_matrix_scaled.shape[0], format='csr')

data_list = [length_matrix_scaled, lanes_matrix_scaled]

network_struct = ModelDataStruct(data_list, incidence_mat)

print("Length range:", sp.csr_matrix.min(length_matrix_scaled), sp.csr_matrix.max(length_matrix_scaled))
print("Lanes range:", sp.csr_matrix.min(lanes_matrix_scaled), sp.csr_matrix.max(lanes_matrix_scaled))

print("Number of zero elements in length matrix:", (length_matrix_scaled.data == 0).sum())
print("Number of zero elements in lanes matrix:", (lanes_matrix_scaled.data == 0).sum())

def test_beta_values(network_struct, beta_range):
    for i in beta_range:
        beta_sim = np.array([-i, -i])  # Using the same value for both betas
        try:
            model = RecursiveLogitModelPrediction(network_struct, initial_beta=beta_sim, mu=1)
            exp_utility = model.get_exponential_utility_matrix()
            if sp.isspmatrix(exp_utility):
                min_val = sp.csr_matrix.min(exp_utility)
                max_val = sp.csr_matrix.max(exp_utility)
            else:
                min_val = np.min(exp_utility)
                max_val = np.max(exp_utility)
            print(f"Beta: {beta_sim}, Exp utility range: {min_val:.6e}, {max_val:.6e}")
            if sp.isspmatrix(exp_utility):
                print(f"Number of zero elements in exp_utility: {(exp_utility.data == 0).sum()}")
            else:
                print(f"Number of zero elements in exp_utility: {(exp_utility == 0).sum()}")
            if min_val > 1e-10 and max_val < 1e10:  # Arbitrary thresholds
                return beta_sim
        except ValueError as e:
            print(f"Error for beta {beta_sim}: {str(e)}")
    return None

# Test a range of beta values
beta_range = np.logspace(-5, 1, 30)  # Test from 1e-5 to 10
best_beta = test_beta_values(network_struct, beta_range)

print("Network statistics:")
print(f"Number of nodes: {network_struct.num_nodes}")
print(f"Number of non-zero elements in incidence matrix: {incidence_mat.nnz}")
print(f"Number of non-zero elements in length matrix: {length_matrix_scaled.nnz}")
print(f"Number of non-zero elements in lanes matrix: {lanes_matrix_scaled.nnz}")

if best_beta is None:
    print("Could not find suitable beta values. Please check your data.")
else:
    print(f"Using beta values: {best_beta}")
    
    # Use the found beta values
    model = RecursiveLogitModelPrediction(network_struct, initial_beta=best_beta, mu=1)
    
    # Generate observations
    num_nodes = network_struct.num_nodes
    orig_indices = np.arange(0, num_nodes, 2)
    dest_indices = (orig_indices + 5) % num_nodes
    obs_per_pair = 1
    obs = model.generate_observations(origin_indices=orig_indices, dest_indices=dest_indices,
                                      num_obs_per_pair=obs_per_pair, iter_cap=2000, rng_seed=42)

    # Estimate the model
    optimiser = optimisers.ScipyOptimiser(method='l-bfgs-b')
    beta_est_init = best_beta / 2  # Start with half of the best beta values
    model_est = RecursiveLogitModelEstimation(network_struct, observations_record=obs,
                                              initial_beta=beta_est_init, mu=1,
                                              optimiser=optimiser)

    beta_est = model_est.solve_for_optimal_beta(verbose=True)

    print(f"beta expected: [{best_beta[0]:6.4f}, {best_beta[1]:6.4f}],"
          f" beta_actual: [{beta_est[0]:6.4f}, {beta_est[1]:6.4f}]")