import numpy as np

from recursiveRouteChoice.data_loading import load_tntp_node_formulation
from recursiveRouteChoice import RecursiveLogitModelPrediction, ModelDataStruct, \
    RecursiveLogitModelEstimation, optimisers

np.set_printoptions(edgeitems=10, linewidth=300)
# np.core.arrayprint._line_width = 500

# DATA
network_file = "SiouxFalls_net.tntp"

data_list, data_list_names = load_tntp_node_formulation(network_file,
                                                        columns_to_extract=["length", 'capacity'])
# print(arc_to_index_map)
distances, capacity = data_list

incidence_mat = (distances > 0).astype(int)

network_struct = ModelDataStruct(data_list, incidence_mat,
                                 data_array_names_debug=("distances", "u_turn"))

beta_vec = np.array([-0.5, -0.1])
model = RecursiveLogitModelPrediction(network_struct,
                                      initial_beta=beta_vec, mu=1)
orig_indices = np.arange(0, 24, 8)
dest_indices = np.arange(0, 24, 8)
obs_per_pair = 4


print(f"Generating {obs_per_pair * len(orig_indices) * len(dest_indices)} obs total per "
      f"configuration")


def get_data(beta_vec, seed=None):

    model = RecursiveLogitModelPrediction(network_struct,
                                          initial_beta=beta_vec, mu=1)
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
n =0
for b1 in np.arange(-0.1, -2, -0.2):
    for b2 in  np.arange(-0.01, -0.1, -0.05):
        n+=1

        beta_gen = np.array([b1, b2])
        try:
            obs = get_data(beta_gen, seed=2)
        except ValueError as e:
            print(f"beta = {beta_gen} failed, {e}")
            continue
        print(obs)
        beta_init = [-0.001, -500]
        model = RecursiveLogitModelEstimation(network_struct, observations_record=obs,
                                              initial_beta=beta_init, mu=1,
                                              optimiser=optimiser)
        beta = model.solve_for_optimal_beta(verbose=False)
        print("beta_expected", beta_gen, "beta actual", beta)

b = time.time()
print("elapsed =", b-a, "s")
