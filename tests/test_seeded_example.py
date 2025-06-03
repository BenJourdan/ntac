import ntac
import os
from os.path import join

from ntac import Ntac

# first download the flywire data we'll need for this example. This will be cached in ~/.ntac/flywire_data
ntac.download_flywire_data(verbose=True)


# SeededNtac is happy to accept an aribtrary CSR adjacency matrix and a list of labels
# where ? corresponds to unlabeled nodes. In this example we will use 
# the flywire dataset, for which we have a custom data loader.

# later we will show how to plug in your own CSR and labels.

# This is where the data lives:
base_path = join(os.path.expanduser("~"), ".ntac")


# now we decide which part of the flywire dataset we want to use:
side = "left"
gender = "female"
area = "ol_columnar"

# below are some other areas:
# area = "ol_intrinsic"
# area = "ol_columnar"
# area = "entire_visual_system"
# area ="central_brain" 
# area = "entire_brain"

ignore_side = False
if area == "central_brain" or area == "entire_brain":
    side = "left_and_right"
if area == "entire_brain_and_nerve_cord":
    side = "left_and_right"
    ignore_side = True


data_path = join(base_path,f"flywire_data/dynamic_data/{gender}_brain/{area}")
edges_file = f"{data_path}/{side}_edges.csv"
types_file = f"{data_path}/{side}_clusters.csv"

print("Loading flywire data")



data = ntac.seeded.FAFBData(edges_file=edges_file, types_file=types_file, ignore_side=ignore_side)

#%%
random_seed = 5

train_indices, test_indices = data.test_train_split(train_size=0.05, sampling_type="at_least_one_per_class", random_seed=random_seed)
# train_indices, test_indices = data.test_train_split(train_size=0.95, sampling_type="uniform", random_seed=random_seed)
#train_indices, test_indices = data.test_train_split(train_size=0.07, sampling_type="exactly_k_per_class", num_per_class=5)
#train_indices, test_indices = data.test_train_split(train_size=0.07, sampling_type="stratified")

#Artifically set test labels to "?"
labels = data.labels.copy()
labels[test_indices] = data.unlabeled_symbol



nt = Ntac(data = data, labels = None, lr=1, topk=1, verbose=True)

nt.solve_unseeded(max_k=10, max_iterations=5)
final_partition = nt.get_partition()
vis = ntac.Visualizer(nt, data)
metrics = data.get_metrics(final_partition, test_indices, data.labels)
vis.plot_acc_vs_class_size(metrics, test_indices=test_indices)




nt = Ntac(data = data, labels = labels, lr=1, topk=1, verbose=True)
num_iters = 15
for i in range(num_iters):
    print(f"Step {i}")
    nt.step()
    final_partition = nt.get_partition()
    metrics = data.get_metrics(final_partition, test_indices, data.labels, compute_class_acc= (i == num_iters - 1))
    print(f"Metrics: {metrics}")


vis = ntac.Visualizer(nt, data)
vis.plot_acc_vs_class_size(metrics, test_indices=test_indices)



# To run seeded Ntac on a realistic example where we don't have all the labels, we only need to pass 
# SeededNtac a scipy CSR adjacency matrix and a list of labels, where the labels are "?" for unlabeled nodes.

adj_csr = data.adj_csr
labels = labels

# Now we can pass SeededNtac the adjacency matrix and labels
nt = Ntac(data = adj_csr, labels =labels, lr=1, topk=1, verbose=True)

# like the previous example, we can run the algorithm for a number of iterations
num_iters = 15
for i in range(num_iters):
    print(f"Step {i}")
    nt.step()

final_partition = nt.get_partition()
print("Final partition for first 10 nodes:", final_partition[:10])