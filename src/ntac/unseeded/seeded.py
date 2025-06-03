from .util import Partition, new_vectors
import numpy as np
import datetime

import concurrent.futures

# build a little one‐arg function that “remembers” both features & median_function
def make_compute_median(feats, median_fn):
    def _compute(cl):
        return median_fn(feats[cl], axis=0)
    return _compute

def medians_from_features(features, clusters, median_function):
    compute_median = make_compute_median(features, median_function)
    with concurrent.futures.ThreadPoolExecutor() as exec:
        medians_list = list(exec.map(compute_median, clusters))
    return np.vstack(medians_list)

def features_and_medians(problem, partition, good_clusters):
    features = problem.new_vectors(partition)
    return features, medians_from_features(features, good_clusters, problem.median_function)

def trusted_medians(problem, partition, old_labels, centers):
    n = problem.numv
    V = range(n)
    set_seeds = set().union(*centers)
    features = problem.new_vectors(partition)

    # Compute good vertices: those that are in the same cluster as in old_labels. Make sure the seeds are always good.
    labels = partition.labels()
    good = [ [] for i in range(partition.size()) ]
    for u in V:
        if labels[u] == old_labels[u] or u in set_seeds:
            good[labels[u]].append(u)
    assert all(len(x) > 0 for x in good)

    # Compute medians using only good vertices; compute distances
    medians = medians_from_features(features, good, problem.median_function)
    distances = problem.all_distances_func(features, medians)

    # Next label assignment; make sure seeds stay in their cluster if there are several seeds per cluster
    next_labels = np.argmin(distances, axis=1)
    for (i, cl) in enumerate(centers):
        for u in cl:
            next_labels[u] = i

    return next_labels

def solve_seeded(problem, centers, max_iterations=12, verbose=False):
    print("Solving seeded!", datetime.datetime.now(), "agg =", problem.median_function, "iterations =", max_iterations)
    k = len(centers)

    old_labels = [ 0 ] * problem.numv               # First step: all in cluster 0, except possibly the seeds
    for (i, cl) in enumerate(centers):
        for u in cl:
            old_labels[u] = i
    partition = Partition(labels = old_labels)
    assert k == partition.size()

    for iterations in range(max_iterations):
        next_labels = trusted_medians(problem, partition, old_labels, centers)
        old_labels = partition.labels().copy()
        partition = Partition(labels=next_labels)

    print("Done!", datetime.datetime.now())
    return partition

# Alternatively, we could use standard NTAC for the seeded problem. We need to be careful here because it requires explicit labels, and we do not want to
# give it the true labels of the seeds, because several of our seeds could belong to the same cluster, but the unseeded algo can't know that.

# The metrics output by data are meaningless now as they don't look for the best match with the reference solution;
# but we can use problem.eval_metrics to the result of this function
# from .convert import partition_from_nt
# from ..seeded import SeededNtac


# def solve_seeded_alternative(problem, centers, max_iterations=12, lr=0.3, verbose=True):
#     data = problem.data
#     labels = np.zeros_like(data.labels)
#     unlabeled = list(set(range(problem.numv)) - set().union(*centers))
#     labels[unlabeled] = data.unlabeled_symbol
#     for (i, cl) in enumerate(centers):
#         for u in cl:
#             labels[u] = problem.cluster_names[i]

#     print("Learning rate =", lr)
#     nt = SeededNtac(data = data, labels =labels, lr=lr, topk=1, verbose=verbose)
#     for i in range(max_iterations):
#         if verbose: print(f"Step {i}")
#         nt.step()
#     partition = partition_from_nt(nt, data)
#     return partition
