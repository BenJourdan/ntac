import numpy as np

import time
from numba import njit, prange
import scipy.sparse as sp
from ..graph_data import GraphData

from .. import unseeded

import sys, os
from contextlib import contextmanager

@contextmanager
def _suppress_stdout():
    """Suppress prints inside the with-block."""
    devnull = open(os.devnull, 'w')
    old_stdout = sys.stdout
    sys.stdout = devnull
    try:
        yield
    finally:
        sys.stdout = old_stdout
        devnull.close()

class Ntac:
    def __init__(self, data, labels = None, lr=0.3, topk=1, verbose=False):
        """
        Initialize the supervised ntac model.

        Parameters:
        - data: Either a scipy sparse CSR matrix or an instance of GraphData.
        - labels: Optional; array of node labels. Required if `data` is a sparse matrix.
        - lr: Learning rate for embedding updates.
        - topk: Number of neighbors to consider in majority voting.
        - verbose: If True, print debugging information.
        """
        self.lr = lr
        self.topk = topk
        self.verbose = verbose
        self.initialized = False
        
        # Check if data is of type csr_matrix; convert to GraphData if needed.
        type_error = "data should be of the type csr_matrix or graph_data"
        if sp.issparse(data):
            labels = np.array(["?"] * data.shape[0]) if labels is None else labels
            try:
                data = GraphData(adj_csr=data, labels=labels)
            except Exception as e:
                print(type_error, e)
                raise
        else:
            assert isinstance(data, GraphData), type_error
            if labels is None:
                labels = data.labels

        self.data = data

        
        unique_labels = np.unique(labels)
        valid_labels = [lbl for lbl in unique_labels  if lbl != data.unlabeled_symbol]
        

        # Create the mapping for valid labels starting at 0.

        label_mapping = {lbl: i for i, lbl in enumerate(valid_labels)}
        label_mapping[data.unlabeled_symbol] = -1
        self.label_mapping = label_mapping
        self.k = len(valid_labels)
        self.reverse_mapping = {new: old for old, new in label_mapping.items()}
        self.labels = np.array([label_mapping[p] if p != data.unlabeled_symbol else -1 for p in labels])
        self.similarity_matrix_to_frozen = None
    

    def solve_unseeded(self, max_k, center_size=5, output_name=None, info_step=1, max_iterations=12, frac_seeds=0.1, chunk_size=6000):
        partition = None # forcing to None for now, because I dont want to expose the Partition class to the user. Can add later
        problem = unseeded.convert.problem_from_data(self.data) #move to init?
        #this crashes when some of the labels are "?"
        if self.verbose:
            best_partition, last_partition, best_history = unseeded.solve_unseeded(problem, max_k, partition, center_size, output_name, info_step, max_iterations, frac_seeds, chunk_size)
        else:
            with _suppress_stdout():
                best_partition, last_partition, best_history = unseeded.solve_unseeded(problem, max_k, partition, center_size, output_name, info_step, max_iterations, frac_seeds, chunk_size)
        partition = best_partition
        # if self.k> 0:
        #     labels = np.array(problem.match_refsol(partition).labels())
        #     # we need to see how we do this when we don't actually get labels in the problem
        #     named_labels = np.array([ problem.cluster_names[labels[u]] for u in range(problem.numv) ])
        #     self.partition = np.array([self.label_mapping[x] for x in named_labels])
        # else:
        self.partition = partition.labels()
        self.solved_unseeded = True
        self.initialized = False
        self.embedding = problem.new_vectors(partition)
        #TODO: add top5 partition here
   
    def _initialize(self):
        """
        Initializes internal variables and computes the initial embedding, similarity matrix, and partition.
        
        Sets up timing metrics, computes reverse mappings for incremental updates, builds the initial embedding
        and similarity matrix between all nodes and the frozen (labeled) nodes, and initializes the partition.
        """
        if self.initialized:
            return
        self.solved_unseeded = False
        self.time_spent_on_embedding = 0.0
        self.time_spent_on_sim_matrix = 0.0
        self.time_spent_on_partition = 0.0

        data = self.data
        csr_indptr = data.adj_csr.indptr
        csr_indices = data.adj_csr.indices
        csc_indptr = data.adj_csc.indptr
        csc_indices = data.adj_csc.indices
        csc_data = data.adj_csc.data
        csr_data = data.adj_csr.data

        # Precompute reverse mappings for incremental updates:
        self.pre_out_indptr, self.pre_out_indices, self.pre_out_data = self._compute_reverse_mapping(csr_indptr, csr_indices, csr_data,data.n)
        self.pre_in_indptr, self.pre_in_indices, self.pre_in_data   = self._compute_reverse_mapping(csc_indptr, csc_indices, csc_data, data.n)

        partition = np.zeros(data.n, dtype=int)
        #set frozen nodes to their labels
        
        #frozen indices are all labeled nodes in data.labels
        frozen_indices = np.where(self.labels != -1)[0]
        frozen_indices = np.sort(frozen_indices)#This is critical for the similarity matrix to be correct
        frozen_partition = self.labels[frozen_indices]
        frozen_percentage = len(frozen_indices) / data.n * 100
        if self.verbose:
            #print("num unique clusters in frozen: ", len(np.unique(frozen_partition)), "out of total clusters: ", self.k)
            print(f"Percentage of frozen vertices: {frozen_percentage:.2f}%, Number of frozen vertices: {len(frozen_indices)}")
       
        partition[frozen_indices] = frozen_partition
        #_t = time.time()
        self.embedding = self._generate_embedding(csr_indptr, csr_indices, csr_data, csc_indptr, csc_indices, csc_data, partition, self.k)
        #self.time_spent_on_embedding += time.time() - _t
        #_t = time.time()
        self.similarity_matrix_to_frozen = self._generate_similarity_matrix_to_frozen(self.embedding, frozen_indices)
        #self.time_spent_on_sim_matrix += time.time() - _t
        
        
        self.partition = partition
        self.frozen_indices = frozen_indices
        self.frozen_partition = frozen_partition
        self.initialized = True
        self.last_embedding = self.embedding.copy()

    @staticmethod
    @njit(parallel=True)
    def _compute_similarity_matrix_sparse_inverted(emb_data, emb_indices, emb_indptr,
                                                    fro_data, fro_indices, fro_indptr,
                                                    row_sum, fro_row_sum,
                                                    N, M, D, tol=1e-9):
        """
        Compute the similarity matrix between full embeddings (N rows) and frozen embeddings (M rows)
        using an inverted-index approach.
        
        The similarity is computed as:
            sim(i,j) = (sum_d min(a_{i,d}, b_{j,d})) / (row_sum[i] + fro_row_sum[j] - sum_d min(a_{i,d}, b_{j,d]))
        
        Parameters:
            emb_data, emb_indices, emb_indptr: Arrays representing the full embeddings in CSC format.
            fro_data, fro_indices, fro_indptr: Arrays representing the frozen embeddings in CSC format.
            row_sum: Precomputed 1D array (length N) with the row sums of the full embeddings.
            fro_row_sum: Precomputed 1D array (length M) with the row sums of the frozen embeddings.
            N: Number of rows in the full embeddings.
            M: Number of rows in the frozen embeddings.
            D: Number of features (columns).
            tol: Tolerance to avoid division by zero.
        
        Returns:
            sim_matrix (np.ndarray): A (N x M) similarity matrix.
        """
        # Allocate dense accumulator for the per-pair intersection sum (sum_d min(...))
        inter = np.zeros((N, M), dtype=np.float32)
        
        # Loop over each feature (column) in parallel.
        for d in prange(D):
            start_full = emb_indptr[d]
            end_full = emb_indptr[d + 1]
            start_fro = fro_indptr[d]
            end_fro = fro_indptr[d + 1]
            # For every nonzero in the full embeddings for this feature:
            for idx_full in range(start_full, end_full):
                i = emb_indices[idx_full]
                a_val = emb_data[idx_full]
                # For every nonzero in the frozen embeddings for this feature:
                for idx_fro in range(start_fro, end_fro):
                    j = fro_indices[idx_fro]
                    b_val = fro_data[idx_fro]
                    # Add the minimum value (which is the contribution of this feature)
                    if a_val < b_val:
                        inter[i, j] += a_val #not a race condition, numba knows how to handle this
                    else:
                        inter[i, j] += b_val

        # Now compute the final similarity matrix using the precomputed row sums.
        sim_matrix = np.zeros((N, M), dtype=np.float32)
        for i in prange(N):
            for j in range(M):
                denom = row_sum[i] + fro_row_sum[j] - inter[i, j]
                if denom < tol:
                    sim_matrix[i, j] = 0.0
                else:
                    sim_matrix[i, j] = inter[i, j] / denom
        return sim_matrix



    def _generate_similarity_matrix_to_frozen(self, embeddings, frozen_indices):
        """
        Generate the similarity matrix between all embeddings and the frozen embeddings.
        
        Parameters:
            embeddings (np.ndarray): A matrix of shape (N x D) representing the embeddings.
            frozen_indices (array-like): Indices of frozen (labeled) nodes.
        
        Returns:
            sim_matrix (np.ndarray): A (N x M) similarity matrix, where M = number of frozen nodes.
        """

        # Build the full embeddings in CSR format.
        embeddings_sparse = sp.csr_matrix(embeddings)
        # Extract the frozen embeddings (this keeps the CSR format).
        frozen_sparse = embeddings_sparse[frozen_indices]
        
        N = embeddings_sparse.shape[0]
        M = frozen_sparse.shape[0]
        # Convert both matrices to CSC format for efficient column access.
        embeddings_csc = embeddings_sparse.tocsc()
        frozen_csc = frozen_sparse.tocsc()
        
        # Get the underlying CSC arrays.
        emb_data = embeddings_csc.data.astype(np.float32)
        emb_indices = embeddings_csc.indices.astype(np.int32)
        emb_indptr = embeddings_csc.indptr.astype(np.int32)
        
        fro_data = frozen_csc.data.astype(np.float32)
        fro_indices = frozen_csc.indices.astype(np.int32)
        fro_indptr = frozen_csc.indptr.astype(np.int32)
        
        # Number of features (columns)
        D = emb_indptr.shape[0] - 1
        
        # Precompute row sums from the CSR representation.
        # (Row sum = sum of nonzero entries for each row.)
        row_sum = np.empty(N, dtype=np.float32)
        for i in range(N):
            row_sum[i] = embeddings_sparse[i].sum()
        
        fro_row_sum = np.empty(M, dtype=np.float32)
        for i in range(M):
            fro_row_sum[i] = frozen_sparse[i].sum()
        
        # Call the inverted-index Numba kernel.
        sim_matrix = self._compute_similarity_matrix_sparse_inverted(
            emb_data, emb_indices, emb_indptr,
            fro_data, fro_indices, fro_indptr,
            row_sum, fro_row_sum,
            N, M, D
        )
    
        return sim_matrix



    @staticmethod
    @njit(parallel=True)
    def _generate_embedding(csr_indptr, csr_indices,csr_data, csc_indptr, csc_indices, csc_data,
                                partition, k):
        """
        Create a 2k-dimensional embedding per vertex.
        
        The embedding is structured as:
            - First k dimensions: in-degree counts per type.
            - Next k dimensions: out-degree counts per type.
        
        Parameters:
            csr_indptr, csr_indices, csr_data: Arrays from the CSR matrix representing out-neighbors.
            csc_indptr, csc_indices, csc_data: Arrays from the CSC matrix representing in-neighbors.
            partition (np.ndarray): Array of vertex types (integers in 0...k-1).
            k (int): Number of distinct types.
        
        Returns:
            embeddings (np.ndarray): A (n x 2k) array representing the computed embeddings.
        """
        n = partition.shape[0]
        t = 2 * k
        embeddings = np.zeros((n, t), dtype=np.float64)
        
        # First pass: compute the in- and out- degree counts.
        for i in prange(n):
            # in-degrees (CSC: nonzeros in column i)
            for j in range(csc_indptr[i], csc_indptr[i+1]):
                p = partition[csc_indices[j]]
                embeddings[i, p] += csc_data[j]
            # out-degrees (CSR: nonzeros in row i)
            for j in range(csr_indptr[i], csr_indptr[i+1]):
                p = partition[csr_indices[j]]
                embeddings[i, k + p] += csr_data[j]


        return embeddings


    @staticmethod
    @njit
    def _topk_indices(similarities, k):
        """
        Returns the indices of the top k elements in `similarities` in descending order.
        
        Uses a partial selection sort algorithm.
        
        Parameters:
            similarities (np.ndarray): A 1D array of similarity scores.
            k (int): The number of top elements to return.
        
        Returns:
            top_indices (np.ndarray): Array of indices corresponding to the top k elements.
        """
        n = similarities.shape[0]
        # Initialize the first k indices and values.
        top_indices = np.empty(k, dtype=np.int64)
        top_values = np.empty(k, dtype=similarities.dtype)
        for i in range(k):
            top_indices[i] = i
            top_values[i] = similarities[i]
        # Sort the initial k elements in descending order.
        for i in range(k):
            for j in range(i + 1, k):
                if top_values[j] > top_values[i]:
                    temp = top_values[i]
                    top_values[i] = top_values[j]
                    top_values[j] = temp
                    temp_idx = top_indices[i]
                    top_indices[i] = top_indices[j]
                    top_indices[j] = temp_idx
        # Process the remaining elements.
        for i in range(k, n):
            val = similarities[i]
            if val > top_values[k - 1]:
                # Find where to insert the new value.
                j = k - 1
                while j >= 0 and val > top_values[j]:
                    j -= 1
                j += 1
                # Shift down to make room.
                for p in range(k - 1, j, -1):
                    top_values[p] = top_values[p - 1]
                    top_indices[p] = top_indices[p - 1]
                top_values[j] = val
                top_indices[j] = i
        return top_indices



    def _compute_partition_from_frozen(self, frozen_indices, frozen_partition, similarity_matrix_to_frozen):
        """
        Computes a partition for non-frozen points based on the top-k closest frozen points.
        
        For frozen points, the known labels are maintained. For non-frozen points, the label is assigned
        via majority voting among the top-k most similar frozen points.
        
        Parameters:
            frozen_indices (np.ndarray): Indices of frozen (labeled) points.
            frozen_partition (np.ndarray): Labels of the frozen points.
            similarity_matrix_to_frozen (np.ndarray): Precomputed similarity matrix (similarities between all points and frozen points).
        
        Returns:
            partition (np.ndarray): The updated partition for all points.
        """
        n = similarity_matrix_to_frozen.shape[0]
        partition = -np.ones(n, dtype=int)  # Initialize all labels to -1 (unassigned)

        # Assign frozen points their known labels #TODO:do we need this?
        partition[frozen_indices] = frozen_partition

        # For non-frozen points, assign the label based on the top c closest frozen points
        non_frozen_indices = np.setdiff1d(np.arange(n), frozen_indices)
        for idx in non_frozen_indices:
            # Get similarities to all frozen points for the current point
            similarities = similarity_matrix_to_frozen[idx]
            if self.topk == 1:
                
                closest_frozen_idx = np.argmax(similarity_matrix_to_frozen[idx])
                closest_frozen_point = frozen_indices[closest_frozen_idx]
                partition[idx] = partition[closest_frozen_point]
            else:
                # Get indices of the top c closest frozen points
                top_c_indices = self._topk_indices(similarities, self.topk)#np.argsort(similarities)[-c:][::-1]  # Descending order

                # Get the labels of these top c frozen points
                top_c_labels = frozen_partition[top_c_indices]

                weights = similarities[top_c_indices]
                epsilon = 1e-5  # to avoid division by zero when s is very close to 1
                # Compute the weight for each neighbor using log-odds weighting.
                weighted_votes = np.log(1 + weights / (1.0 - weights + epsilon))
                label_scores = np.zeros(np.max(top_c_labels) + 1)
                for label, vote in zip(top_c_labels, weighted_votes):
                    label_scores[label] += vote
                assigned_label = np.argmax(label_scores)

                # Assign the determined label to the current point
                partition[idx] = assigned_label

        return partition




    def _compute_reverse_mapping(self, csr_indptr, csr_indices, csr_data, n):
        """
        Computes the reverse mapping for a sparse matrix.
        
        For each node, computes how many times it appears as a neighbor in the original CSR matrix,
        and then builds arrays representing the reverse mapping (i.e., for each node, the list of nodes that point to it).
        
        Parameters:
            csr_indptr (np.ndarray): CSR index pointer array.
            csr_indices (np.ndarray): CSR indices array.
            csr_data (np.ndarray): CSR data array (weights).
            n (int): Number of nodes.
        
        Returns:
            tuple: A tuple containing:
                - rev_indptr (np.ndarray): Pointer array for the reverse mapping.
                - rev_indices (np.ndarray): Indices array for the reverse mapping.
                - rev_data (np.ndarray): Data array (weights) for the reverse mapping.
        """
        # First, count how many times each node appears as a neighbor.
        counts = np.zeros(n, dtype=np.int32)
        for i in range(n):
            for idx in range(csr_indptr[i], csr_indptr[i+1]):
                node = csr_indices[idx]
                counts[node] += 1

        # Build the pointer array for the reverse mapping.
        rev_indptr = np.empty(n+1, dtype=np.int32)
        rev_indptr[0] = 0
        for i in range(n):
            rev_indptr[i+1] = rev_indptr[i] + counts[i]

        # Allocate arrays to hold the reverse mapping indices and the corresponding weights.
        rev_indices = np.empty(rev_indptr[-1], dtype=np.int32)
        rev_data = np.empty(rev_indptr[-1], dtype=csr_data.dtype)

        # Temporary counter array to keep track of insertion positions.
        temp = np.zeros(n, dtype=np.int32)
        for i in range(n):
            for idx in range(csr_indptr[i], csr_indptr[i+1]):
                node = csr_indices[idx]
                pos = rev_indptr[node] + temp[node]
                rev_indices[pos] = i
                rev_data[pos] = csr_data[idx]
                temp[node] += 1

        return rev_indptr, rev_indices, rev_data



    @staticmethod
    @njit#(parallel=True)#surprisingly, this is function is faster without parallelization
    def _incremental_embedding_update(embedding, 
                                    pre_in_indptr, pre_in_indices, pre_in_data,
                                    pre_out_indptr, pre_out_indices, pre_out_data,
                                    old_partition, new_partition, 
                                    k, changed_nodes):
        """
        Incrementally update the embeddings for all nodes affected by changes in partition assignments.
        
        For each changed node j, the function adjusts the embedding of all its in-neighbors and out-neighbors
        by subtracting the contribution of the old label and adding that of the new label.
        
        Parameters:
            embedding (np.ndarray): The current embedding matrix.
            pre_in_indptr, pre_in_indices, pre_in_data: Reverse mapping arrays for in-neighbors.
            pre_out_indptr, pre_out_indices, pre_out_data: Reverse mapping arrays for out-neighbors.
            old_partition (np.ndarray): The partition labels before the update.
            new_partition (np.ndarray): The updated partition labels.
            k (int): Number of classes (used to index the embedding dimensions).
            changed_nodes (np.ndarray): Array of node indices whose labels have changed.
        """
        # For each node j that changed its partition:
        
        #commented out for parallelization, since it is not faster
        #also, the function is much faster than computing the sim matrix
        # so no need to optimize it for now

        # num_changed = changed_nodes.size
        # for j_idx in prange(num_changed):
        #     j = changed_nodes[j_idx]
        
        for j in changed_nodes:
            old_label = old_partition[j]
            new_label = new_partition[j]
            # Update all nodes i that have j as an in-neighbor.
            start = pre_in_indptr[j]
            end = pre_in_indptr[j+1]
            for idx in range(start, end):
                i = pre_in_indices[idx]
                weight = pre_in_data[idx]  # Weight of edge i <- j.
                if new_label != -1:
                    embedding[i, new_label] +=  weight
                if old_label != -1:
                    embedding[i, old_label] -=  weight

            # Update all nodes i that have j as an out-neighbor.
            start = pre_out_indptr[j]
            end = pre_out_indptr[j+1]
            for idx in range(start, end):
                i = pre_out_indices[idx]
                weight = pre_out_data[idx]  # Weight of edge i <- j in the out-neighbor context.
                # Update the out-neighbor part (columns k to 2*k-1):
                if new_label != -1:
                    embedding[i, k + new_label] += weight
                if old_label != -1:
                    embedding[i, k + old_label] -= weight



    def step(self):
        """
        Performs one step of the seeded ntac algorithm.
        
        This includes:
            1. Updating the partition from the similarity matrix.
            2. Computing the incremental embedding update for nodes whose partitions have changed.
            3. Blending the updated embedding with the previous state using the learning rate.
            4. Recomputing the similarity matrix to frozen nodes.
        
        Also updates internal timing metrics for different operations.
        """
        if not self.initialized:
            self._initialize()
        _t = time.time()
        old_partition = self.partition.copy()
        self.partition = self._compute_partition_from_frozen(self.frozen_indices, self.frozen_partition, self.similarity_matrix_to_frozen)

        changed_nodes = np.nonzero(self.partition != old_partition)[0]
        # print("Percentage of nodes that changed partition: ", changed_nodes.size / len(partition))

        self.time_spent_on_partition += time.time() - _t

        _t = time.time()

        # last_embedding = self.generate_embedding(self.data.adj_csr.indptr, self.data.adj_csr.indices, self.data.adj_csr.data,
        #                                 self.data.adj_csc.indptr, self.data.adj_csc.indices, self.data.adj_csc.data,
        #                                 self.partition, self.k)
        if changed_nodes.size > 0:
            self._incremental_embedding_update(self.last_embedding, 
                            self.pre_in_indptr, self.pre_in_indices, self.pre_in_data,
                            self.pre_out_indptr, self.pre_out_indices, self.pre_out_data,
                            old_partition, self.partition, 
                            self.k, changed_nodes)
        self.embedding = (1-self.lr) * self.embedding + self.lr * self.last_embedding

        
        self.time_spent_on_embedding += time.time() - _t
        
        _t = time.time()
        self.similarity_matrix_to_frozen = self._generate_similarity_matrix_to_frozen(self.embedding, self.frozen_indices)
        self.time_spent_on_sim_matrix += time.time() - _t

        if self.verbose:
            # print(f"Time spent on initial embedding: {self.time_spent_on_initial_embedding:.2f} seconds")
            # print(f"Time spent on initial similarity matrix: {self.time_spent_on_initial_sim:.2f} seconds")
            print(f"Time spent on embedding: {self.time_spent_on_embedding:.2f} seconds")
            print(f"Time spent on similarity matrix update: {self.time_spent_on_sim_matrix:.2f} seconds")
            print(f"Time spent on partition: {self.time_spent_on_partition:.2f} seconds")
        



    def get_partition(self):
        """
        Returns the current partition after remapping the numeric labels back to the original labels.
        
        Returns:
            np.ndarray: Array of labels corresponding to the current partition.
        """
       
        if self.solved_unseeded:
            # If the unseeded problem was solved, return the partition directly
            
            return np.array([str(p) for p in self.partition])
        if not self.initialized:
            #throw an erroe asking the use to run the alg first
            raise RuntimeError("Need to run at least one step or call solve_unseeded() before getting the partition.")
        return np.array([self.reverse_mapping[p] for p in self.partition])



    def _build_class_ptrs(self):
        """
        Build flattened arrays mapping each class to its frozen-column indices in the similarity matrix.
        Populates:
          - self._fro_cols: int32[R] array of all frozen-column indices grouped by class
          - self._fro_ptr:  int32[K+1] array of pointers into _fro_cols
        """
        K = self.k
        # Initialize pointer array
        fro_ptr = np.zeros(K + 1, dtype=np.int32)
        cols = []
        idx = 0
        # For each class, collect the indices of frozen_partition equal to that class
        for c in range(K):
            # mark the start of class c
            fro_ptr[c] = idx
            # find columns j where frozen_partition[j] == c
            for j, cls in enumerate(self.frozen_partition):
                if cls == c:
                    cols.append(j)
                    idx += 1
        # final pointer
        fro_ptr[K] = idx
        # store
        self._fro_cols = np.array(cols, dtype=np.int32)
        self._fro_ptr  = fro_ptr

    @staticmethod
    @njit(parallel=True)
    def _topk_njit(sim, fro_cols, fro_ptr, topk, K):
        """
        Numba kernel: for each of n rows in sim (shape n×M), compute the top-k distinct classes:
        - sim[i, j]: similarity of node i to frozen-column j
        - fro_cols, fro_ptr define which j's belong to which class
        Returns:
          topk_inds: int32[n×topk] of class indices
          topk_vals: float32[n×topk] of their max-sim values
        """
        n, M = sim.shape
        # outputs
        topk_inds = np.empty((n, topk), dtype=np.int32)
        topk_vals = np.empty((n, topk), dtype=np.float32)

        # scratch per-class array
        for i in prange(n):
            # 1) compute max similarity per class
            class_scores = np.zeros(K, dtype=np.float32)
            for c in range(K):
                start = fro_ptr[c]
                end   = fro_ptr[c+1]
                maxv = 0.0
                for idx in range(start, end):
                    j = fro_cols[idx]
                    v = sim[i, j]
                    if v > maxv:
                        maxv = v
                class_scores[c] = maxv

            # 2) partial-sort to find topk classes
            # initialize
            for t in range(topk):
                topk_inds[i, t] = t
                topk_vals[i, t] = class_scores[t]
            # sort initial window
            for a in range(topk):
                for b in range(a+1, topk):
                    if topk_vals[i, b] > topk_vals[i, a]:
                        tmpv = topk_vals[i, a]; topk_vals[i, a] = topk_vals[i, b]; topk_vals[i, b] = tmpv
                        tmpc = topk_inds[i, a]; topk_inds[i, a] = topk_inds[i, b]; topk_inds[i, b] = tmpc
            # process rest
            for c in range(topk, K):
                v = class_scores[c]
                if v > topk_vals[i, topk-1]:
                    # find insertion point
                    pos = topk-1
                    while pos >= 0 and v > topk_vals[i, pos]:
                        pos -= 1
                    pos += 1
                    # shift down
                    for s in range(topk-1, pos, -1):
                        topk_vals[i, s] = topk_vals[i, s-1]
                        topk_inds[i, s] = topk_inds[i, s-1]
                    topk_vals[i, pos] = v
                    topk_inds[i, pos] = c

        return topk_inds, topk_vals
    
    def get_topk_partition(self,  k=None):
        """
        Returns the top-k partition labels and their similarity scores for each node,
        formatted as a dict mapping node_index → list of (label, similarity) tuples.

        Parameters:
            k (int): Number of top labels to return. 

        Returns:
            Dict[int, List[Tuple[label, float]]]: A dictionary where each key is a node index (0…n-1)
            and each value is a list of `k` tuples `(original_label, similarity_score)`, sorted by descending score.
        """
        if self.solved_unseeded:
            exception_msg = "Cannot call get_topk_partition() after solving the unseeded problem. Use get_partition() instead."
            raise RuntimeError(exception_msg)
        if not self.initialized:
            #throw an erroe asking the use to run the alg first
            raise RuntimeError("Need to run at least one step before getting the partition.")
        # Ensure class pointers are built
        if not hasattr(self, "_fro_cols"):
            self._build_class_ptrs()

        # Cast similarity matrix to float32 for the Numba kernel

        sim = self.similarity_matrix_to_frozen.astype(np.float32)

        # Run the Numba kernel: returns (n × topk) arrays of class‐indices and their max‐similarity scores
        topk_inds, topk_vals = Ntac._topk_njit(
            sim,
            self._fro_cols,
            self._fro_ptr,
            k,
            self.k
        )

        n = sim.shape[0]
        result = {}

        # Build a dict mapping node → [(label, score), …]
        for i in range(n):
            tuples = []
            for j in range(k):
                num_label = int(topk_inds[i, j])
                if self.solve_unseeded:
                    # If the unseeded problem was solved, we can directly use the partition
                    original_label = num_label
                original_label = self.reverse_mapping[num_label]
                score = float(topk_vals[i, j])
                tuples.append((original_label, score))
            result[i] = tuples

        return result


    def map_partition_to_gt_labels(self, gt_labels):
        #TODO: fix docstring
        """
        Given:
        - partition: 1D array of predicted labels (strings), length N
        - self.labels:  1D array of ground-truth labels (strings), length N

        This uses only util functions (labels2clusters, match_clusters, cluster_labels)
        to align predicted clusters to ground-truth clusters and return a 1D array
        of matched GT‐label strings.
        """

        from ntac.unseeded.util import labels2clusters, match_clusters, cluster_labels

        # 1) Encode predicted‐label strings → integer codes in [0..P−1]
        pred_arr = np.asarray(self.partition, dtype=object)
        pred_tokens, pred_inv = np.unique(pred_arr, return_inverse=True)
        # Now pred_inv[i] is an integer in [0..P−1] for node i

        # 2) Encode GT‐label strings → integer codes in [0..G−1]
        gt_arr = np.asarray(gt_labels, dtype=object)
        gt_tokens, gt_inv = np.unique(gt_arr, return_inverse=True)
        # Now gt_inv[i] is an integer in [0..G−1] for node i

        # 3) Build list-of-clusters from those integer codes
        clusters_pred = labels2clusters(pred_inv)
        clusters_gt   = labels2clusters(gt_inv)

        # 4) Run the matcher; C[j] = list of node‐indices from the predicted cluster
        #    that best matches GT cluster j
        C = match_clusters(clusters_pred, clusters_gt)

        # 5) Flatten C back into an integer‐label array of length N, where each node u
        #    gets assigned to GT‐cluster‐index j if u ∈ C[j]
        matched_int = cluster_labels(C)
        # matched_int[i] is now in [0..G−1]

        # 6) Convert those ints back to the original GT‐strings
        #    gt_tokens[j] is the string name of GT cluster j
        self.partition = np.array(gt_tokens[matched_int], dtype=object)