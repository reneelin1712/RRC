import pandas as pd
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
import networkx as nx
import matplotlib.pyplot as plt
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