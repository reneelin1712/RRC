import numpy as np

from recursiveRouteChoice.data_loading import load_tntp_to_sparse_arc_formulation

np.set_printoptions(edgeitems=10, linewidth=300)
from recursiveRouteChoice import (ModelDataStruct, RecursiveLogitModelPrediction,
                                  RecursiveLogitModelEstimation, optimisers)
# testing replacements
# from recursiveRouteChoice import (ModelDataStruct, RecursiveLogitModelPredictionSM as
# RecursiveLogitModelPrediction, RecursiveLogitModelEstimationSM as RecursiveLogitModelEstimation,
#                                   optimisers)



# np.core.arrayprint._line_width = 500

# DATA
# network_file = "SiouxFalls_net.tntp"
network_file = "EMA_net.tntp"
# network_file = "berlin-mitte-center_net.tntp"
# network_file = "berlin-prenzlauerberg-center_net.tntp"
arc_to_index_map, distances = load_tntp_to_sparse_arc_formulation(network_file, columns_to_extract=["length"],
                                                                  )
# print(arc_to_index_map)
index_node_pair_map = {v: k for (k, v) in arc_to_index_map.items()}

# distances = np.array(
#     [[4, 3.5, 4.5, 3, 3, 0, 0, 0],
#      [3.5, 3, 4, 0, 2.5, 3, 3, 0],
#      [4.5, 4, 5, 0, 0, 0, 4, 3.5],
#      [3, 0, 0, 2, 2, 2.5, 0, 2],
#      [3, 2.5, 0, 2, 2, 2.5, 2.5, 0],
#      [0, 3, 0, 2.5, 2.5, 3, 3, 2.5],
#      [0, 3, 4, 0, 2.5, 3, 3, 2.5],
#      [0, 0, 3.5, 2, 0, 2.5, 2.5, 2]])
#

incidence_mat = (distances > 0).astype(int)

data_list = [distances]
print("max dist = ", np.max(distances.A))
network_struct = ModelDataStruct(data_list, incidence_mat,
                                 data_array_names_debug=("distances", "u_turn"))

beta_vec = np.array([-1])
model = RecursiveLogitModelPrediction(network_struct,
                                      initial_beta=beta_vec, mu=1)
print("Linear system size", model.get_exponential_utility_matrix().shape)
# orig_indices = np.arange(1, 240, 8)
# dest_indices = np.arange(2, 240, 8)
orig_indices = np.linspace(1, 240, 20).astype(int)
dest_indices = np.linspace(2, 240, 80).astype(int)
# orig_indices = np.arange(0, 7, 1)
# dest_indices = np.arange(0, 7, 1)
obs_per_pair = 2


print(f"Generating {obs_per_pair * len(orig_indices) * len(dest_indices)} obs total per "
      f"configuration")


def get_data(beta, seed=None):
    beta_vec_generate = np.array([beta])
    model = RecursiveLogitModelPrediction(network_struct,
                                          initial_beta=beta_vec_generate, mu=1)
    obs = model.generate_observations(origin_indices=orig_indices,
                                      dest_indices=dest_indices,
                                      num_obs_per_pair=obs_per_pair, iter_cap=2000, rng_seed=seed,
                                      )
    return obs


# =======================================================
print(120 * "=", 'redo with scipy')
optimiser = optimisers.ScipyOptimiser(method='l-bfgs-b')  # bfgs, l-bfgs-b


import time
a = time.time()
for n, beta_gen in enumerate(np.array([-1.0, -5]), start=1):
    try:
        obs = get_data(beta_gen, seed=2)
    except ValueError as e:
        print(f"beta = {beta_gen} failed, {e}")
        continue
    # print(obs)
    beta_init = -5
    model = RecursiveLogitModelEstimation(network_struct, observations_record=obs,
                                          initial_beta=beta_init, mu=1,
                                          optimiser=optimiser)
    beta = model.solve_for_optimal_beta(verbose=False)
    print("beta_expected", beta_gen, "beta actual", beta)

b = time.time()
print("elapsed =", b-a, "s")
