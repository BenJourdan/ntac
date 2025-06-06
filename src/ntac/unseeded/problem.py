from .util import Partition, new_vectors, new_vectors_sparse, match_clusters, compare_clusters, weighted_jaccard_distance
from .utilgpu import weighted_jaccard_cpu_fast, weighted_jaccard_gpu_csr, is_cuda_available
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.linalg import norm
from sklearn.metrics import adjusted_rand_score, accuracy_score, f1_score
from math import log, ceil, floor
import pandas as pd

from bottleneck import nanmean
from bottleneck import median as bnmedian

class Problem:
    def __init__(self, edges, vertex_names, partition, name, cluster_names=None):
        self.edges = edges
        self.vertex_names = np.array(vertex_names)
        self.id2num_ = None
        self.int_edges_ = None
        self.numv = len(self.vertex_names)
        self.refsol = partition
        self.A_ = None
        self.directed = True
        self.name = name
        self.cluster_names = cluster_names

        self.median_function = nanmean
#        self.median_function = bnmedian
        self.distance_measure = weighted_jaccard_distance

        self.log_file = ""
        self.sols_dir = "./sols/"
        self.seeded_method = 0

        self.set_device(verbose=True)

    def set_device(self, device="default", verbose=False):
        if device=="cpu" or not is_cuda_available():
            if verbose: print("No CUDA, using CPU")
            self.all_distances_func = weighted_jaccard_cpu_fast
        else:
            if verbose: print("Using CUDA")
            self.all_distances_func = weighted_jaccard_gpu_csr

    def id2num_map(self):
        if self.id2num_ is None:
            self.id2num_ = {}
            for (i, u) in enumerate(self.vertex_names):
                self.id2num_[u] = i
        return self.id2num_

    def id2num(self, name):
        return self.id2num_map()[name]

    def int_edges(self):
        if self.int_edges_ is None:
            self.int_edges_ = [ (self.id2num(u), self.id2num(v), w) for (u, v, w) in self.edges ]
        return self.int_edges_

    def induced_subgraph(self, subset):
        new_names = self.vertex_names[subset]
        subset = set(subset)
        new_edges = [ (u, v, w) for (u, v, w) in self.edges if self.id2num(u) in subset and self.id2num(v) in subset ]
        new_clusters = [ list(subset & set(c)) for c in self.refsol.clusters()]
        new_clusters = [ c for c in new_clusters if len(c) > 0 ]

        ret = Problem(new_edges, new_names, None, self.name, None)
        ret.refsol = Partition(clusters=[ [ret.id2num(self.vertex_names[u]) for u in c] for c in new_clusters ])
        return ret

    # private, only to be called once
    def adj_matrix(self):
        edges = self.int_edges()

        # Extract row indices (vertex1), column indices (vertex2), and weights
        rows = np.array([edge[0] for edge in edges])
        cols = np.array([edge[1] for edge in edges])
        data = np.array([edge[2] for edge in edges])

        # Create a sparse matrix in COO format
        return coo_matrix((data, (rows, cols)), shape=(self.numv, self.numv)).toarray()

    # Adjacency matrix in compressed row format
    def A(self):
        if self.A_ is None:
            self.A_ = csr_matrix(self.adj_matrix())
        return self.A_

    def new_vectors(self, partition):
        return new_vectors(self.A(), partition.clusters(), self.directed)

    def new_vectors_sparse(self, partition):
        return new_vectors_sparse(self.A(), partition.clusters(), self.directed)

    def eval_metrics(self, partition, metrics=['clusters','l1', 'ml1', 'l2', 'jac', 'mis'], compute_class_acc=False):
        labels = partition.labels()
        clusters = partition.clusters()
        d = dict()
        if 'clusters' in metrics:
            d['clusters'] = max(labels) + 1

        vec = self.new_vectors_sparse(partition)
        avg = csr_matrix([bnmedian(vec[cl].toarray(), axis=0) for cl in clusters])
        ext_avg = avg[labels]
        if 'l1' in metrics:
            diffl1 = norm(vec - ext_avg, 1, axis=1)
            d['l1'] = diffl1.sum()
            d['ml1'] = diffl1.max()
        if 'jac' in metrics:            
            d['jac'] = 1 - np.average(vec.minimum(ext_avg).sum(axis=1) / vec.maximum(ext_avg).sum(axis=1))
        if 'l2' in metrics:            
            averages = csr_matrix([nanmean(vec[cl].toarray(), axis=0) for cl in clusters])
            d['l2'] = norm(vec - averages[labels], 'fro')

        if self.refsol is not None and 'mis' in metrics and partition.size() <= self.refsol.size():
            q = self.match_refsol(partition)
            d["ari"] = adjusted_rand_score(self.refsol.labels(), q.labels())
            d["f1"] = f1_score(self.refsol.labels(), q.labels(), average='weighted')
            d["acc"] = accuracy_score(self.refsol.labels(), q.labels())
            if compute_class_acc:
                d["class_acc"] = { self.cluster_names[i]:len(set(cl) & set(q.clusters()[i])) / len(cl) for (i, cl) in enumerate(self.refsol.clusters()) if i < len(q.clusters()) }
        return d

    def match_refsol(self, partition):
        return Partition(clusters=match_clusters(partition.clusters(), self.refsol.clusters()))

    def correct_fraction_labels(self, partition):
        correct = compare_clusters(partition.clusters(), self.refsol.clusters())
        return correct / self.numv

    def add_solution_file(self, types_file, verbose=True):
        types_df = pd.read_csv(types_file)
        types_df = types_df[types_df['vertex'].isin(self.vertex_names)]

        nodes = types_df['vertex']
        sorted_names = sorted(nodes, key=lambda x: (len(x), x))
        node_to_idx = {node: idx for idx, node in enumerate(sorted_names)}
        n = len(nodes)
        assert n == self.numv

        # Find a string that doesn't exist in the column
        missing = types_df['cluster'].isna().sum()
        existing_values = set(types_df['cluster'].dropna().unique())
        # Create a replacement string that's not in the existing values
        replacement = "Unknown"
        counter = 1
        while replacement in existing_values:
            replacement = f"Unknown_{counter}"
            counter += 1

        # Replace NaN values with the unique replacement string
        types_df['cluster'] = types_df['cluster'].fillna(replacement)
        if missing > 0:
            print("Missing %i labels! Setting a new cluster %s" % (missing, replacement))

        ground_truth_partition = np.full(n, '', dtype=object)
        for _, row_data in types_df.iterrows():
            vertex = row_data['vertex']
            if vertex in node_to_idx:
                idx = node_to_idx[vertex]
                ground_truth_partition[idx] = row_data['cluster']

        # === 4. Return the results ===
        # (You can later use `top_regions` and `cluster_capacities` to filter neurons whose top region
        # does not match the expected capacity and region for their cluster.)
        node_to_neuron_id = {}
        for _, row_data in types_df.iterrows():
            vertex = row_data["vertex"]
            neuron_id = row_data["neuron id"]
            if vertex in node_to_idx:
                node_to_neuron_id[vertex] = neuron_id

        vertex_names = list(node_to_neuron_id.keys())
        unique_labels = types_df['cluster'].unique()

        label_map = {x:i for (i, x) in enumerate(unique_labels)}
        gt_labels = [ label_map[ground_truth_partition[u]] for u in range(n) ]
        vertex_names = list(node_to_neuron_id.keys())
        self.vertex_names = vertex_names
        self.refsol = Partition(labels=gt_labels)
        self.cluster_names = unique_labels

        if verbose:
            print("reference solution:", types_file)
            print(self.eval_metrics(self.refsol), "\n")

    def save_csv(self, partition, output_file, match_refsol=False):
        q = partition.clean_empty()
        vertices = self.vertex_names
        if match_refsol:
            q = self.match_refsol(q)
            clusters = [ self.cluster_names[ q.labels()[v] ] for v in range(self.numv) ]
        else:
            clusters = [ "c%i" % (q.labels()[v] + 1) for v in range(self.numv) ]
        df = pd.DataFrame({'vertex': vertices, 'cluster': clusters})
        df_sorted = df.sort_values(by='vertex', key=lambda x: x.map(lambda s: (len(s), s)))
        df_sorted.to_csv(output_file, index=False)
        print(f"Saved to '{output_file}'.")
