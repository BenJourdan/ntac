# from ..seeded import SeededNtac
from ..graph_data import FAFBData


from .problem import Problem
from .util import Partition, fp_type
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

def read_input(edges_file, types_file):
    print("Loading flywire data")
    data = FAFBData(edges_file=edges_file, types_file=types_file)
    problem = problem_from_data(data, name=types_file)
    print("problem and reference solution:", edges_file, types_file)
    print("n =", problem.numv, "m =", len(problem.A().nonzero()[0]))
    print(problem.eval_metrics(problem.refsol), "\n")
    return problem, data

# Takes a Problem and a Data and returns an Ntac object
# def convert_to_nt(problem, partition, data):
#     labels = np.array(problem.match_refsol(partition).labels())
#     named_labels = np.array([ problem.cluster_names[labels[u]] for u in range(problem.numv) ])
#     nt = SeededNtac(data = data, labels = named_labels, lr=0.3, topk=1, verbose=True)
#     nt.partition = np.array([nt.label_mapping[x] for x in named_labels])
#     nt.embedding = problem.new_vectors(partition)
#     assert np.all(nt.get_partition() == named_labels)
#     return nt

# Takes a Data and returns a Problem
def problem_from_data(data, name=""):
    coo = data.adj_csr.tocoo()
    edges = list(zip(coo.row, coo.col, coo.data))
    # label_map = {x:i for (i, x) in enumerate(data.unique_labels)}
    # label_names = data.unique_labels.copy()
    # if isinstance(data, FAFBData):
    #     if '?' in np.unique(data.ground_truth_partition):
    #         assert not ('?' in label_names)
    #         label_names = list(label_names) + ['?']
    #         label_map['?'] = len(label_names) - 1
    #     gt_labels = [ label_map[ data.ground_truth_partition[ u ] ] for u in range(data.n) ]

    #     vertex_names = [ data.idx_to_node[u] for u in range(data.n) ]
    #     problem = Problem([(vertex_names[u], vertex_names[v], w) for (u, v, w) in edges ], vertex_names, Partition(labels=gt_labels), name, data.unique_labels)
    #     problem.A_ = data.adj_csr.astype(fp_type)                        # no need to recompute it, slow
    #     problem.data = data
    # else:
        # If no ground truth partition is available, we create a trivial partition
    #get nodes fron edges
    nodes = np.unique([str(u) for (u, _, _) in edges] + [str(v) for (_, v, _) in edges])
    sorted_names = sorted(nodes, key=lambda x: (len(x), x))
    #node_to_idx = {node: idx for idx, node in enumerate(sorted_names)}
    vertex_names = sorted_names
    problem = Problem([(vertex_names[u], vertex_names[v], w) for (u, v, w) in edges ], vertex_names, Partition(labels=[0]*data.n), name=name, cluster_names=["trivial_cluster"])
    problem.A_ = data.adj_csr.astype(fp_type)                        # no need to recompute it, slow
    problem.refsol = None
        
    return problem
# Takes Ntac and Data and returns a Partition
def partition_from_nt(nt, data):
    return Partition(labels=nt.partition)

# Read input without type (cluster) information
def read_from_graph_only(edges_file):
    # === 1. Load all data without filtering ===
    edges_df = pd.read_csv(edges_file)

    # Get the full set of nodes from the edges file (do not filter yet)
    nodes = set(pd.unique(edges_df[['from', 'to']].values.ravel('K')))

    sorted_names = sorted(nodes, key=lambda x: (len(x), x))
    # Build a mapping from node name to index (sorting ensures deterministic ordering).
    node_to_idx = {node: idx for idx, node in enumerate(sorted_names)}
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}
    n = len(nodes)

    # Build the initial adjacency matrix using all nodes.
    row = edges_df['from'].map(node_to_idx).values
    col = edges_df['to'].map(node_to_idx).values
    data = edges_df["weight"].values
    adj_csr = csr_matrix((data, (row, col)), shape=(n, n))

    coo = adj_csr.tocoo()
    edges = list(zip(coo.row, coo.col, coo.data))
    vertex_names = sorted_names
    problem = Problem([(vertex_names[u], vertex_names[v], w) for (u, v, w) in edges ], vertex_names, Partition(labels=[0]*n), name=edges_file, cluster_names=["trivial_cluster"])
    problem.A_ = adj_csr.astype(fp_type)                        # no need to recompute it, slow
    problem.refsol = None

    print("n =", problem.numv, "m =", len(problem.A().nonzero()[0]))
    return problem
