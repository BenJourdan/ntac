import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix

from bottleneck import nanmean
from bottleneck import median as bnmedian

from numba import jit, njit, prange           # jit compilation
from copy import copy
import pickle

from scipy.sparse import csr_matrix, hstack, vstack
from joblib import Parallel, delayed

#fp_type = np.float64
fp_type = np.float32

class Partition:
    def __init__(self, *, labels=None, clusters=None):
        self.labels_ = None
        self.clusters_ = None
        if labels is not None: self.labels_ = np.copy(labels)
        elif clusters is not None: self.clusters_ = copy(clusters)
        else:
            assert False

    def labels(self):
        if self.labels_ is None: self.labels_ = cluster_labels(self.clusters_)
        return self.labels_

    def clusters(self):
        if self.clusters_ is None: self.clusters_ = labels2clusters(self.labels_)
        return self.clusters_

    def size(self):
        return len(self.clusters())

    def clean_empty(self):
        return Partition(clusters=[c for c in self.clusters() if len(c) > 0])

def cluster_labels(clusters):
    n = max([max(c, default=-1) for c in clusters]) + 1
    labels = [ -1 ] * n
    for (i, c) in enumerate(clusters):
        for u in c:
            labels[u] = i
    return labels

def labels2clusters(labels):
    ret = [ [] for i in range( max(labels) + 1) ]
    for (u, l) in enumerate(labels):
        ret[l].append(u)
    return ret

def all_intersections(A, B):

    # 1) build a global vocab mapping each token → column-index
    vocab = {}
    rowsA, colsA = [], []
    for i,a in enumerate(A):
        for x in set(a):
            colsA.append(vocab.setdefault(x, len(vocab)))
            rowsA.append(i)
    dataA = [1]*len(rowsA)
    A_mat = csr_matrix((dataA, (rowsA, colsA)),
                       shape=(len(A), len(vocab)))

    # 2) same for B
    rowsB, colsB = [], []
    for j,b in enumerate(B):
        for x in set(b):
            if x in vocab:
                colsB.append(vocab[x])
                rowsB.append(j)
    dataB = [1]*len(rowsB)
    B_mat = csr_matrix((dataB, (rowsB, colsB)),
                       shape=(len(B), len(vocab)))

    # 3) now the all-pairs intersection sizes:
    return (A_mat @ B_mat.T).toarray()

# B is the reference, A is the solution found
def match_clusters(A, B):
    if len(A) < len(B):
        A = A + [ [] ] * (len(B) - len(A))

    score = all_intersections(A, B)
    row_indices, col_indices = linear_sum_assignment(score, maximize=True)
    best_score = score[row_indices, col_indices].sum()

    C = [ [] for i in range(len(row_indices)) ]

    for (i, j) in zip(row_indices, col_indices):
        C[j] = A[i]
    return C

def compare_clusters(A, B):
    if len(A) <= len(B):
        C = match_clusters(A, B)
        correct_labels = sum(np.array(cluster_labels(C)) == np.array( cluster_labels(B)))
        return correct_labels
    else:
        return -1

# Old computation of feature vectors using CPU
def new_vectors_cpu(A, clusters, directed):
    points = np.column_stack([A[:, c].sum(axis=1).A1 for c in clusters])
    if directed:
        points2 = np.column_stack([A.T[:, c].sum(axis=1).A1 for c in clusters])
        return np.hstack([points, points2])
    else:
        return points

@njit(parallel=True)
def _new_vectors_numba(n, k, indptr, indices, data,
                       indptr_T, indices_T, data_T,
                       cluster_map, directed):
    # output has k cols (undirected) or 2k cols (directed)
    out = np.zeros((n, k * (2 if directed else 1)), dtype=data.dtype)
    for i in prange(n):
        # outgoing sums
        for p in range(indptr[i], indptr[i+1]):
            j = indices[p]
            c = cluster_map[j]
            out[i, c] += data[p]
        if directed:
            # incoming sums (via A.T)
            for p in range(indptr_T[i], indptr_T[i+1]):
                j = indices_T[p]
                c = cluster_map[j]
                out[i, k + c] += data_T[p]
    return out

def new_vectors(A, clusters, directed):
    # 1) ensure CSR and grab its arrays
    A_csr = A.tocsr()
    indptr, indices, data = A_csr.indptr, A_csr.indices, A_csr.data

    # 2) same for transpose
    AT_csr = A_csr.T.tocsr()
    indptr_T, indices_T, data_T = AT_csr.indptr, AT_csr.indices, AT_csr.data

    # 3) flatten clusters → array
    n = A_csr.shape[0]
    cluster_map = cluster_labels(clusters)
    cluster_map = np.array(cluster_map, dtype=np.int32)     # avoid type reflection warning

    # 4) call the fast numba kernel
    k = len(clusters)
    return _new_vectors_numba(n, k,
                              indptr, indices, data,
                              indptr_T, indices_T, data_T,
                              cluster_map,
                              1 if directed else 0)

def new_vectors_sparse(A, clusters, directed,
                         n_jobs=-1, chunk_size=None):
    n = A.shape[0]
    # build S as before
    cmap = np.asarray(cluster_labels(clusters), dtype=np.int32)
    S = csr_matrix((np.ones(n, A.dtype),
                    (np.arange(n), cmap)),
                   shape=(n, len(clusters)))

    # decide chunking
    if chunk_size is None:
        # about 4–8 chunks per core
        chunk_size = max(1, n // (8 * abs(n_jobs)))
    row_chunks = np.array_split(np.arange(n), 
                                np.ceil(n/chunk_size).astype(int))

    # worker for outgoing
    def out_chunk(rows):
        return A[rows, :].dot(S)

    # parallel compute outgoing
    outs = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(out_chunk)(chunk) for chunk in row_chunks
    )
    out = vstack(outs, format="csr")

    if directed:
        AT = A.T.tocsr()
        def inc_chunk(rows):
            return AT[rows, :].dot(S)
        incs = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(inc_chunk)(chunk) for chunk in row_chunks
        )
        inc = vstack(incs, format="csr")
        out = hstack([out, inc], format="csr")

    return out

@njit
def weighted_jaccard_distance(a, b):
    M = np.maximum(a, b).sum()
    return np.linalg.norm(a - b, 1) / M if M != 0 else 1

def describe_sol(problem, p, verbose=False, reorder=True, max_show=200):
    show_max = 20

    if reorder:
        cl = sorted(p.clusters(), key=lambda x:-len(x))
    else:
        cl = p.clusters()
    print(problem.eval_metrics(p))

    if problem.refsol is not None:
        summary = []
        for (i, x) in enumerate(cl):
            lab = [problem.refsol.labels()[u] for u in x]
            unique_vals, counts = np.unique(lab, return_counts=True)
            sort_indices = np.argsort(-counts)
            top_values = [problem.cluster_names[x] for x in unique_vals[sort_indices[:show_max]]]
            top_counts = counts[sort_indices[:show_max]]

            summary.append( (top_values[0], top_counts[0], len(x) ) )
            if verbose: print("i =", i, "size =", len(x), list(zip(top_values, top_counts)))
        print("main clusters detected: main_type (correct, total) =", [ "%s (%i/%i)" % (label, correct, total) for (label, correct, total) in summary ][:max_show])
        distinct_labels = list(dict.fromkeys( [label for (label, _, _) in summary] ))
        print("distinct labels =", len(distinct_labels), "%.3lg%%" % (len(distinct_labels) / p.size() * 100))

        refinement_score = 0
        for (i, x) in enumerate(problem.refsol.clusters()):
            lab = [p.labels()[u] for u in x]
            unique_vals, counts = np.unique(lab, return_counts=True)
            sort_indices = np.argsort(-counts)

            top_values = unique_vals[sort_indices]
            top_counts = counts[sort_indices]
            score = top_counts[0]

            refinement_score += score
        print("refinement score =", refinement_score, refinement_score / len(p.labels()))
    else:
        print("sizes =", sorted([ len(x) for x in p.clusters() ], key=lambda x:-x))

def precomputed_solution(filename):
    print("loading", filename)
    with open(filename, "rb") as infile:
        centers, labels = pickle.load(infile)
    return Partition(labels=labels), centers

