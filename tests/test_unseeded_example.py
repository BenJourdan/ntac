import ntac
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

problem, data = ntac.unseeded.read_input(edges_file, types_file)    

best_partition, last_partition, history = ntac.unseeded.solve_unseeded(problem, problem.refsol.size(), partition=None, output_name="test2")
print(problem.eval_metrics(last_partition, compute_class_acc=True), "\n")
print(problem.eval_metrics(best_partition, compute_class_acc=True), "\n")

# We could also examine some of the stored intermediate solutions
#from util import precomputed_solution
#partition, centers = precomputed_solution("sols/unseeded_example_5_%i.pickle" % 25)

# Convert my solution to labels expected by UnseededNtac
nt = ntac.unseeded.convert_to_nt(problem, best_partition, data)
metrics = data.get_metrics(nt.get_partition(), np.array(range(problem.numv)), data.labels, compute_class_acc=True)
print(f"Metrics: {metrics}")

vis = ntac.Visualizer(nt, data)
vis.plot_class_accuracy(metrics)
vis.plot_embedding_comparison("R7", "R8", show_error=False)
vis.plot_true_label_histogram("T4a", top_k=5)
vis.plot_confusion_matrix(normalize=True, fscore_threshold=0.95)
problematic_labels = ["R7", "R8", "T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]
vis.plot_confusion_matrix(normalize=True, include_labels=problematic_labels)

