import pandas as pd
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
import networkx as nx
import matplotlib.pyplot as plt
import ast
from recursiveRouteChoice import ModelDataStruct, RecursiveLogitModelPrediction, RecursiveLogitModelEstimation
from recursiveRouteChoice import optimisers

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
        lanes_matrix[from_node, to_node] = parse_lanes(edge['lanes'])

# Convert matrices to sparse format
incidence_mat = sparse.csr_matrix(adjacency_matrix)
lanes_matrix = sparse.csr_matrix(lanes_matrix)

# Check connectivity
n_components, labels = connected_components(csgraph=incidence_mat, directed=True, return_labels=True)
print(f"Number of strongly connected components: {n_components}")

if n_components > 1:
    component_sizes = np.bincount(labels)
    largest_component = np.argmax(component_sizes)
    print(f"Component sizes: {component_sizes}")
    print(f"Largest component size: {component_sizes[largest_component]}")
    mask = labels == largest_component
    incidence_mat = incidence_mat[mask][:, mask]
    lanes_matrix = lanes_matrix[mask][:, mask]
    num_nodes = incidence_mat.shape[0]
    print(f"Using largest connected component with {num_nodes} nodes")

# Print detailed statistics about the matrices
print("\nIncidence matrix:")
print("Shape:", incidence_mat.shape)
print("Non-zero elements:", incidence_mat.nnz)
print("Density:", incidence_mat.nnz / (incidence_mat.shape[0] * incidence_mat.shape[1]))

print("\nLanes matrix:")
print("Shape:", lanes_matrix.shape)
print("Non-zero elements:", lanes_matrix.nnz)
if sparse.isspmatrix_dok(lanes_matrix):
    data = np.array(list(lanes_matrix.values()))
else:
    data = lanes_matrix.data
print("Min:", data.min())
print("Max:", data.max())
print("Mean:", data.mean())
print("Median:", np.median(data))

# Normalize lanes matrix
lanes_matrix = lanes_matrix / data.max()

data_list = [lanes_matrix]
network_struct = ModelDataStruct(data_list, incidence_mat)

def find_suitable_beta(network_struct, mu_range, beta_range):
    def get_matrix_stats(matrix):
        if sparse.isspmatrix_dok(matrix):
            data = np.array(list(matrix.values()))
        else:
            data = matrix.data
        return data.min(), data.max(), data.mean()

    for mu in mu_range:
        for beta in beta_range:
            beta_sim = np.array([beta])
            try:
                model = RecursiveLogitModelPrediction(network_struct, initial_beta=beta_sim, mu=mu)
                
                # Debug: Print the exponential utility matrix
                exp_utility = model.get_exponential_utility_matrix()
                print(f"\nTesting beta={beta}, mu={mu}")
                print(f"Exponential utility matrix stats:")
                min_val, max_val, mean_val = get_matrix_stats(exp_utility)
                print(f"  Min: {min_val}")
                print(f"  Max: {max_val}")
                print(f"  Mean: {mean_val}")
                
                # Try to solve for value functions
                value_functions = model._compute_exp_value_function(exp_utility, model.is_network_data_sparse)
                print(f"Value functions stats:")
                print(f"  Min: {value_functions.min()}")
                print(f"  Max: {value_functions.max()}")
                print(f"  Mean: {value_functions.mean()}")
                
                # If we got here without an exception, try to generate an observation
                obs = model.generate_observations(origin_indices=[0], dest_indices=[1],
                                                  num_obs_per_pair=1, iter_cap=2000)
                print(f"Successfully generated an observation")
                return beta_sim, mu
            except Exception as e:
                print(f"Failed with beta={beta_sim}, mu={mu}. Error: {str(e)}")
                continue
    raise ValueError("Could not find suitable beta values")

# Find suitable beta values
mu_range = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
beta_range = np.concatenate([np.arange(-10.0, -0.1, 0.1), np.arange(-0.1, 0, 0.001)])
try:
    beta_sim, mu = find_suitable_beta(network_struct, mu_range, beta_range)
except ValueError as e:
    print(f"Error: {e}")
    exit(1)

# Generate some observations
num_obs = 100
orig_indices = np.random.randint(0, network_struct.incidence_matrix.shape[0], num_obs)
dest_indices = np.random.randint(0, network_struct.incidence_matrix.shape[0], num_obs)

model = RecursiveLogitModelPrediction(network_struct, initial_beta=beta_sim, mu=mu)
obs = model.generate_observations(origin_indices=orig_indices, dest_indices=dest_indices,
                                  num_obs_per_pair=1, iter_cap=2000, rng_seed=42)

# Estimate the model
optimiser = optimisers.ScipyOptimiser(method='l-bfgs-b')
beta_est_init = beta_sim.copy()

try:
    model_est = RecursiveLogitModelEstimation(network_struct, observations_record=obs,
                                              initial_beta=beta_est_init, mu=mu,
                                              optimiser=optimiser)

    beta_est = model_est.solve_for_optimal_beta(verbose=True)

    print(f"beta initial: [{beta_sim[0]:6.4f}], beta_estimated: [{beta_est[0]:6.4f}]")
except Exception as e:
    print(f"Error in model estimation: {e}")

# Create a NetworkX graph for further analysis
G = nx.from_scipy_sparse_array(incidence_mat, create_using=nx.DiGraph)

print("\nNetwork Analysis:")
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())
print("Is strongly connected:", nx.is_strongly_connected(G))
print("Number of strongly connected components:", nx.number_strongly_connected_components(G))
print("Average in-degree:", sum(dict(G.in_degree()).values()) / G.number_of_nodes())
print("Average out-degree:", sum(dict(G.out_degree()).values()) / G.number_of_nodes())

# Identify nodes with no outgoing edges (sinks)
sinks = [node for node, out_degree in G.out_degree() if out_degree == 0]
print("Number of sink nodes (no outgoing edges):", len(sinks))

# Identify nodes with no incoming edges (sources)
sources = [node for node, in_degree in G.in_degree() if in_degree == 0]
print("Number of source nodes (no incoming edges):", len(sources))

# Plot degree distribution
in_degrees = [d for n, d in G.in_degree()]
out_degrees = [d for n, d in G.out_degree()]

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.hist(in_degrees, bins=20)
plt.title("In-degree Distribution")
plt.xlabel("In-degree")
plt.ylabel("Frequency")

plt.subplot(122)
plt.hist(out_degrees, bins=20)
plt.title("Out-degree Distribution")
plt.xlabel("Out-degree")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig("degree_distribution.png")
plt.close()

# Visualize the network (only if it's small enough)
if G.number_of_nodes() <= 100:
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, node_size=20, node_color='lightblue', 
            with_labels=False, arrows=True)
    plt.title("Network Visualization")
    plt.savefig("network_visualization.png")
    plt.close()
else:
    print("Network too large to visualize entirely. Consider visualizing a subgraph.")