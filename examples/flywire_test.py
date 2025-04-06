"""A simple example of how to use ntac on the flywire dataset."""

import os
import numpy as np
from ntac import GraphData, Ntac

side = "left"
gender = "female"
area = "entire_brain"


ignore_side = False
if area == "central_brain" or area == "entire_brain":
    side = "left_and_right"
if area == "entire_brain_and_nerve_cord":
    side = "left_and_right"
    ignore_side = True

data_path = f"dynamic_data/{gender}_brain/{area}"

edges_file = f"{data_path}/{side}_edges.csv"
types_file = f"{data_path}/{side}_clusters.csv"
dataset = data_path.split("/")[-1] + "_" + side


cache_dir = os.path.join(os.path.expanduser("~"), ".ntac")
base_data_path = os.path.join(
    cache_dir, "flywire_data/dynamic_data/female_brain/entire_visual_system"
)
print(f"Using data from {base_data_path}")
edges_file = f"{base_data_path}/right_edges.csv"
types_file = f"{base_data_path}/right_clusters.csv"
top_regions_summary_file = (
    f"dynamic_data/{gender}_brain/most_common_top_regions_per_type.csv"
)


data = GraphData(
    edges_file=edges_file,
    types_file=types_file,
    ignore_side=ignore_side,
    return_neuron_ids=True,
)


labels = data.ground_truth_partition.copy()
original_labels = labels.copy()

use_test_set = True

if use_test_set:
    labeled_indices = np.flatnonzero(labels != "?")
    test_size = int(np.where(labels != "?")[0].shape[0] * 0.1)
    test_set = np.random.choice(labeled_indices, test_size, replace=False)
    labels[test_set] = "?"


partitions = []


def get_accuracy(partition, test_set, original_labels):
    """Calculate the accuracy of the partition on the test set."""
    return (
        np.where(partition[test_set] == original_labels[test_set])[0].shape[0]
        / test_set.shape[0]
    )


train_labels = labels.copy()

nt = Ntac(rounds=10, ground_truth_frac=0.05, lr=0.3, topk=1, force_all_clusters=True)

partition, frozen_indices, embedding, label_mapping, verification_indices = nt.fit(
    data.adj_csr, train_labels, tentative_labels=None
)
metrics = data.get_metrics(partition, test_set)
print(f"Metrics: {metrics}")
