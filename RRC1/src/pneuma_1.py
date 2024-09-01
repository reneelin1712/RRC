import pandas as pd
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from recursiveRouteChoice import ModelDataStruct, RecursiveLogitModelPrediction, RecursiveLogitModelEstimation
from recursiveRouteChoice import optimisers
import ast

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

# Print detailed statistics about the matrices
print("\nIncidence matrix:")
print("Shape:", incidence_mat.shape)
print("Non-zero elements:", incidence_mat.nnz)
print("Density:", incidence_mat.nnz / (incidence_mat.shape[0] * incidence_mat.shape[1]))

print("\nLength matrix:")
print("Shape:", length_matrix.shape)
print("Non-zero elements:", length_matrix.nnz)
print("Min:", length_matrix.data.min())
print("Max:", length_matrix.data.max())
print("Mean:", length_matrix.data.mean())
print("Median:", np.median(length_matrix.data))

print("\nLanes matrix:")
print("Shape:", lanes_matrix.shape)
print("Non-zero elements:", lanes_matrix.nnz)
print("Min:", lanes_matrix.data.min())
print("Max:", lanes_matrix.data.max())
print("Mean:", lanes_matrix.data.mean())
print("Median:", np.median(lanes_matrix.data))

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

# Function to test a single beta value
def test_beta(network_struct, beta_sim, mu):
    try:
        model = RecursiveLogitModelPrediction(network_struct, initial_beta=beta_sim, mu=mu)
        obs = model.generate_observations(origin_indices=[0], dest_indices=[1],
                                          num_obs_per_pair=1, iter_cap=2000)
        return True
    except Exception as e:
        print(f"Failed with beta={beta_sim}, mu={mu}. Error: {str(e)}")
        return False

# Test a range of beta values
mu = 1.0
beta_range = np.arange(-5.0, 0.1, 0.1)
for beta1 in beta_range:
    for beta2 in beta_range:
        beta_sim = np.array([beta1, beta2])
        if test_beta(network_struct, beta_sim, mu):
            print(f"Suitable beta values found: {beta_sim}, mu: {mu}")
            break
    else:
        continue
    break
else:
    print("Could not find suitable beta values")