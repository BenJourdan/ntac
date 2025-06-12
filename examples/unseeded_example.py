import ntac
from ntac import Ntac, FAFBData
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
area = "ol_intrinsic" #we pick a small area to speed up the example

data_path = join(base_path,f"flywire_data/dynamic_data/{gender}_brain/{area}")
edges_file = f"{data_path}/{side}_edges.csv"
types_file = f"{data_path}/{side}_clusters.csv"

data = FAFBData(edges_file=edges_file, types_file=types_file)

#No need to pass in labels, as we will be using the unseeded method
nt = Ntac(data = data, labels = None, verbose=True)

#print num unique labels
print(f"Number of unique labels: {data.unique_labels.shape[0]}")
nt.solve_unseeded(
    max_k=230
)

nt.map_partition_to_gt_labels(data.labels)
partition = nt.get_partition()
metrics = data.get_metrics(partition, np.array(range(data.n)), data.labels)
print(f"Accuracy: {metrics['acc']:.3f} ARI: {metrics['ari']:.3f}", f"Weighted F1: {metrics['f1']:.3f}")


