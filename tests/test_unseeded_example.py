#%%
import ntac
from ntac import Ntac
import os
from os.path import join
import numpy as np



# first download the flywire data we'll need for this example. This will be cached in ~/.ntac/flywire_data
ntac.download_flywire_data(verbose=True)


# This is where the data lives:
base_path = join(os.path.expanduser("~"), ".ntac")


# now we decide which part of the flywire dataset we want to use:
side = "left"
gender = "female"
area = "ol_columnar"


data_path = join(base_path,f"flywire_data/dynamic_data/{gender}_brain/{area}")
edges_file = f"{data_path}/{side}_edges.csv"
types_file = f"{data_path}/{side}_clusters.csv"

data = ntac.seeded.FAFBData(edges_file=edges_file, types_file=types_file)


#%%

labels = data.labels
#set some of the labels to "?"
labels[labels == "R7"] = "?"
nt = Ntac(data = data, labels = labels, verbose=True)

nt.solve_unseeded(
    max_k=5
)

#%%

metrics = data.get_metrics(nt.get_partition(), np.array(range(data.n)), data.labels, compute_class_acc=True)
print(f"Metrics: {metrics}")
#%%
vis = ntac.Visualizer(nt, data)
vis.plot_class_accuracy(metrics)
vis.plot_embedding_comparison("R7", "R8", show_error=False)
# vis.plot_true_label_histogram("T4a", top_k=5)
vis.plot_confusion_matrix(normalize=True, fscore_threshold=0.95)
problematic_labels = [
    "R7", 
    "R8", 
    # "T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"
    ]
vis.plot_confusion_matrix(normalize=True, include_labels=problematic_labels)

#%%
# Now we show how to use ntac in the unseeded mode, where we start with an adjacency matrix and no labels.

adj_matrix_csr = data.adj_csr

nt = Ntac(data = adj_matrix_csr, labels = None, verbose=True)

nt.solve_unseeded(
    max_k=5
)
partition = nt.get_partition()
partition = data.map_partition_to_gt_labels(partition)
metrics = data.get_metrics(partition, np.array(range(data.n)), data.labels, compute_class_acc=False)
# %%
