from .util import Partition, weighted_jaccard_distance, describe_sol
from .problem import Problem
from .seeded import solve_seeded, features_and_medians
from numba import njit
import math
import os
import datetime, time
import pickle
import numpy as np

def partition_cost(problem, partition):
    return problem.eval_metrics(partition, ['jac'])['jac']

def best_representatives(problem, partition, t):
    features, medians = features_and_medians(problem, partition, partition.clusters())
    distances = np.column_stack([ problem.distance_measure(features[u], medians[partition.labels()[u]]) for u in range(problem.numv) ] ).reshape(-1)
    return [ np.array(cl)[np.argsort(distances[cl])][:min(t, (len(cl) + 1) // 2)] for cl in partition.clusters() ]

def solve_unseeded(problem, max_k, partition=None, center_size=5, output_name=None, info_step=1, max_iterations=12, frac_seeds=0.1, chunk_size=6000):
    start_time = time.perf_counter()

    assert partition is None or partition.size() <= max_k
    if partition is None: partition = Partition(clusters=[range(problem.numv)])

    solve_seeded_method = solve_seeded

    best_history = [ (partition, partition_cost(problem, partition)) ]
    while partition.size() <= max_k:
        # Recompute centers to match better the current partition
        centers = best_representatives(problem, partition, center_size)
        assert centers is None or np.all( (not x.empty() for x in centers) )

        if partition.size() % info_step == 0 or partition.size() == max_k:
            # Show info about the current solution
            print("\n\n============ %s k = %i/%i %s =============" % (problem.name, partition.size(), max_k, problem.log_file), datetime.datetime.now(), "Time elapsed:", "%.2f" % (time.perf_counter() - start_time), "seconds")
            describe_sol(problem, partition)
            print("best so far: ", "size =", best_history[-1][0].size(), "jac =", best_history[-1][1])
            if problem.refsol is not None: print("distinct labels in centers[:,0] =", len(set(np.array([problem.refsol.labels()[c[0]] for c in centers]))))

        # Save the solution
        if output_name is not None:
            folder_name = problem.sols_dir
            # create the directory if it does not exist
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            with open('%s/%s_unseeded_%i_%i.pickle' % (problem.sols_dir, output_name, center_size, partition.size()), 'wb') as outfile:
                print("Saving to", outfile.name)
                pickle.dump((centers, partition.labels()), outfile)

        if partition.size() == max_k: break

        # Add a new seed as new cluster center
        next_seed = get_next_seed(problem, partition, centers, frac_seeds, chunk_size)
        if next_seed is None:
            print("Cant't find a new seed, terminating early")
            break

        centers.append(np.array([next_seed]))
        old_label = partition.labels()[next_seed]
        centers[old_label] = np.array([u for u in centers[old_label] if u != next_seed])     # can't serve as seed for different clusters
        partition = solve_seeded_method(problem, centers, max_iterations=max_iterations)
        jac = partition_cost(problem, partition)

        if jac < best_history[-1][-1]: best_history.append((partition, jac))

    return best_history[-1][0], partition, best_history

@njit(parallel=True)
def calc_gains(distances, new_dists):
    return np.maximum(distances.reshape(-1, 1) - new_dists, 0).sum(axis=0)

def get_next_seed(problem, partition, centers, fraction=1, chunk_size=6000):
    print("Selecting seed...", datetime.datetime.now(), "fraction =", fraction)
    assert 0 <= fraction <= 1
    try_seeds = math.ceil(fraction * problem.numv)

    # Take some candidates from among the vertices furthest way from the centers of their clusters
    features, medians = features_and_medians(problem, partition, centers)
    distances = np.column_stack([ problem.distance_measure(features[u], medians[partition.labels()[u]]) for u in range(problem.numv) ] ).reshape(-1)

    singletons = set().union(*(c for c in centers if len(c) == 1))        # disallow choosing a singleton center as new seed
    candidates_ranked_by_cluster = [ np.array(cl)[np.argsort(-distances[cl])] for (i, cl) in enumerate(partition.clusters()) ]
    choices = []
    i = 0
    k = len(partition.clusters())
    while len(choices) < try_seeds:
        for j in range(k):
            if len(candidates_ranked_by_cluster[j]) > i:
                choices.append( candidates_ranked_by_cluster[j][i] )
        i += 1
    choices = list(set(choices) - singletons)
    if len(choices) == 0: return None
    print("choices =", len(choices))

    # Among the candidates in choices, pick the one with the most immediate Jaccard benefit
    chunks = np.array_split(choices, (len(choices) + chunk_size - 1) // chunk_size)
    if len(chunks) > 1: print("%i chunks, chunk_size = %i" % (len(chunks), len(chunks[0])))
    gains = np.hstack([ calc_gains(distances, problem.all_distances_func(features, features[chunk])) for chunk in chunks ])

    new_seed = choices[np.argmax(gains)]
    print("new_seed =", new_seed, "real label =", problem.cluster_names[problem.refsol.labels()[new_seed]] if problem.refsol is not None else "", "gain =", max(gains), datetime.datetime.now())
    return new_seed
