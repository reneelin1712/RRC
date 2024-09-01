import pandas as pd
import numpy as np
from scipy import sparse
from recursiveRouteChoice import ModelDataStruct, RecursiveLogitModelPrediction, RecursiveLogitModelEstimation
from recursiveRouteChoice import optimisers
import ast

# Read the data
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

# Print original ranges and statistics
print("\nOriginal Length range:", np.min(length_matrix.data), np.max(length_matrix.data))
print("Original Length mean:", np.mean(length_matrix.data))
print("Original Length median:", np.median(length_matrix.data))
print("Original Capacity range:", np.min(lanes_matrix.data), np.max(lanes_matrix.data))
print("Original Capacity mean:", np.mean(lanes_matrix.data))
print("Original Capacity median:", np.median(lanes_matrix.data))

# Scale the matrices
length_scale = np.mean(length_matrix.data)
capacity_scale = np.max(lanes_matrix.data)
length_matrix = length_matrix / length_scale
lanes_matrix = lanes_matrix / capacity_scale

data_list = [length_matrix, lanes_matrix]
network_struct = ModelDataStruct(data_list, incidence_mat)

print("\nScaled Length range:", np.min(length_matrix.data), np.max(length_matrix.data))
print("Scaled Length mean:", np.mean(length_matrix.data))
print("Scaled Length median:", np.median(length_matrix.data))
print("Scaled Capacity range:", np.min(lanes_matrix.data), np.max(lanes_matrix.data))
print("Scaled Capacity mean:", np.mean(lanes_matrix.data))
print("Scaled Capacity median:", np.median(lanes_matrix.data))

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
    elif isinstance(matrix, np.ndarray):
        if matrix.size > 0:
            return matrix.min(), matrix.max()
        else:
            return 0, 0
    else:
        raise ValueError(f"Unsupported matrix type: {type(matrix)}")

def find_suitable_parameters(network_struct, beta_length_range, beta_lanes_range, mu_range):
    for mu in mu_range:
        for beta_length in beta_length_range:
            for beta_lanes in beta_lanes_range:
                try:
                    model = RecursiveLogitModelPrediction(network_struct, initial_beta=np.array([-beta_length, -beta_lanes]), mu=mu)
                    exp_utility = model.get_exponential_utility_matrix()
                    min_val, max_val = get_matrix_range(exp_utility)
                    print(f"Mu: {mu}, Beta_length: {-beta_length}, Beta_lanes: {-beta_lanes}, Exp utility range: {min_val:.6e}, {max_val:.6e}")
                    if min_val > 1e-10 and max_val < 1e10:
                        # Try to generate observations
                        try:
                            orig_indices = np.random.randint(0, num_nodes, 10)
                            dest_indices = np.random.randint(0, num_nodes, 10)
                            model.generate_observations(origin_indices=orig_indices, dest_indices=dest_indices,
                                                        num_obs_per_pair=1, iter_cap=2000, rng_seed=42)
                            return mu, -beta_length, -beta_lanes
                        except ValueError as e:
                            print(f"Failed to generate observations with mu {mu}, beta_length {-beta_length}, and beta_lanes {-beta_lanes}: {str(e)}")
                except ValueError as e:
                    print(f"Error for mu {mu}, beta_length {-beta_length}, and beta_lanes {-beta_lanes}: {str(e)}")
    return None, None, None

# Find suitable parameters
beta_length_range = np.logspace(-4, 2, 30)  # Test from 1e-4 to 100
beta_lanes_range = np.logspace(-4, 2, 30)   # Test from 1e-4 to 100
mu_range = np.logspace(-2, 3, 20)  # Test from 0.01 to 1000
best_mu, best_beta_length, best_beta_lanes = find_suitable_parameters(network_struct, beta_length_range, beta_lanes_range, mu_range)

if best_mu is None or best_beta_length is None or best_beta_lanes is None:
    print("Could not find suitable parameters. Please check your data.")
else:
    print(f"Best mu value found: {best_mu}")
    print(f"Best beta_length value found: {best_beta_length}")
    print(f"Best beta_lanes value found: {best_beta_lanes}")

    # Initialize the model with the best parameter values found
    beta_sim = np.array([best_beta_length, best_beta_lanes])
    model = RecursiveLogitModelPrediction(network_struct, initial_beta=beta_sim, mu=best_mu)

    # Generate some observations
    num_obs = 100
    orig_indices = np.random.randint(0, num_nodes, num_obs)
    dest_indices = np.random.randint(0, num_nodes, num_obs)
    obs = model.generate_observations(origin_indices=orig_indices, dest_indices=dest_indices,
                                      num_obs_per_pair=1, iter_cap=2000, rng_seed=42)

    # Estimate the model
    optimiser = optimisers.ScipyOptimiser(method='l-bfgs-b')
    beta_est_init = [best_beta_length/2, best_beta_lanes/2]  # Start with half of the best beta values
    model_est = RecursiveLogitModelEstimation(network_struct, observations_record=obs,
                                              initial_beta=beta_est_init, mu=best_mu,
                                              optimiser=optimiser)

    beta_est = model_est.solve_for_optimal_beta(verbose=True)

    print(f"beta expected: [{beta_sim[0]:6.4f}, {beta_sim[1]:6.4f}],"
          f" beta_actual: [{beta_est[0]:6.4f}, {beta_est[1]:6.4f}]")

    # Print some statistics about the matrices
    print("\nAdjacency matrix shape:", adjacency_matrix.shape)
    print("Number of non-zero elements in adjacency matrix:", np.count_nonzero(adjacency_matrix))
    print("Length matrix shape:", length_matrix.shape)
    print("Number of non-zero elements in length matrix:", length_matrix.nnz)
    print("Lanes matrix shape:", lanes_matrix.shape)
    print("Number of non-zero elements in lanes matrix:", lanes_matrix.nnz)