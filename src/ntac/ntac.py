"""The NTAC class."""

import random
import time

import numpy as np
import scipy.sparse as sp
from numba import njit, prange
from tqdm import tqdm


class Ntac:
    """NTAC class for the NTAC algorithm."""

    def __init__(
        self,
        rounds,
        ground_truth_frac=0.05,
        lr=0.3,
        topk=1,
        force_all_clusters=True,
        verbose=False,
    ):
        """Setup the NTAC class."""

        self.rounds = rounds
        self.ground_truth_frac = ground_truth_frac
        self.lr = lr
        self.topk = topk
        self.force_all_clusters = force_all_clusters
        self.verbose = verbose

    @staticmethod
    @njit(parallel=True)
    def generate_initial_embedding(csr_indptr, csr_indices, csc_indptr, csc_indices):
        """Generate the initial embedding for each vertex."""

        n = csr_indptr.shape[0] - 1
        embeddings = np.zeros((n, 7), dtype=np.float32)
        self_in_degrees = np.empty(n, dtype=np.int32)
        self_out_degrees = np.empty(n, dtype=np.int32)

        # Compute self in/out degrees (unweighted; equivalent to summing rows/cols)
        for i in prange(n):
            self_in_degrees[i] = csc_indptr[i + 1] - csc_indptr[i]
            self_out_degrees[i] = csr_indptr[i + 1] - csr_indptr[i]

        # For each vertex compute neighbor-average degrees and number of 2-cycles.
        for i in prange(n):
            # In-neighbors: from CSC column i.
            in_start = csc_indptr[i]
            in_end = csc_indptr[i + 1]
            n_in = in_end - in_start
            sum_in_deg_in = 0.0
            sum_out_deg_in = 0.0
            for j in range(in_start, in_end):
                nb = csc_indices[j]
                sum_in_deg_in += self_in_degrees[nb]
                sum_out_deg_in += self_out_degrees[nb]
            avg_in_deg_in = sum_in_deg_in / n_in if n_in > 0 else 0.0
            avg_out_deg_in = sum_out_deg_in / n_in if n_in > 0 else 0.0

            # Out-neighbors: from CSR row i.
            out_start = csr_indptr[i]
            out_end = csr_indptr[i + 1]
            n_out = out_end - out_start
            sum_in_deg_out = 0.0
            sum_out_deg_out = 0.0
            for j in range(out_start, out_end):
                nb = csr_indices[j]
                sum_in_deg_out += self_in_degrees[nb]
                sum_out_deg_out += self_out_degrees[nb]
            avg_in_deg_out = sum_in_deg_out / n_out if n_out > 0 else 0.0
            avg_out_deg_out = sum_out_deg_out / n_out if n_out > 0 else 0.0

            # Number of 2-cycles: count common vertices in in-neighbors and out-neighbors.
            num_2_cycles = 0
            j = in_start
            k_ptr = out_start
            while j < in_end and k_ptr < out_end:
                a = csc_indices[j]
                b = csr_indices[k_ptr]
                if a == b:
                    num_2_cycles += 1
                    j += 1
                    k_ptr += 1
                elif a < b:
                    j += 1
                else:
                    k_ptr += 1

            embeddings[i, 0] = self_in_degrees[i]
            embeddings[i, 1] = self_out_degrees[i]
            embeddings[i, 2] = avg_in_deg_in
            embeddings[i, 3] = avg_out_deg_in
            embeddings[i, 4] = avg_in_deg_out
            embeddings[i, 5] = avg_out_deg_out
            embeddings[i, 6] = num_2_cycles

        return embeddings

    @staticmethod
    @njit(parallel=True)
    def compute_similarity_matrix_sparse_inverted(
        emb_data,
        emb_indices,
        emb_indptr,
        fro_data,
        fro_indices,
        fro_indptr,
        row_sum,
        fro_row_sum,
        N,
        M,
        D,
        tol=1e-9,
    ):
        """Compute the similarity matrix between full embeddings (N rows) and frozen embeddings (M rows) using an inverted-index approach."""

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
                        inter[i, j] += a_val
                    else:
                        inter[i, j] += b_val

        # Now compute the final similarity matrix using the precomputed row sums.
        sim_matrix = np.zeros((N, M), dtype=np.float32)
        for i in prange(N):
            for j in range(M):
                # Denom = row_sum[i] + fro_row_sum[j] - inter[i,j]
                denom = row_sum[i] + fro_row_sum[j] - inter[i, j]
                if denom < tol:
                    sim_matrix[i, j] = 0.0
                else:
                    sim_matrix[i, j] = inter[i, j] / denom
        return sim_matrix

    def generate_similarity_matrix_to_frozen(self, embeddings, frozen_indices):
        """Generate the similarity matrix between all embeddings and the frozen embeddings.

        Uses one of two methods; if method=="inverted" the inverted-index (columnwise) approach is used.
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
        sim_matrix = self.compute_similarity_matrix_sparse_inverted(
            emb_data,
            emb_indices,
            emb_indptr,
            fro_data,
            fro_indices,
            fro_indptr,
            row_sum,
            fro_row_sum,
            N,
            M,
            D,
        )

        return sim_matrix

    @staticmethod
    @njit(parallel=True)
    def generate_embedding(
        csr_indptr,
        csr_indices,
        csr_data,
        csc_indptr,
        csc_indices,
        csc_data,
        partition,
        k,
    ):
        """Create a 2k-dimensional embedding per vertex."""

        n = partition.shape[0]
        t = 2 * k
        embeddings = np.zeros((n, t), dtype=np.float32)

        # First pass: compute the in- and out- degree counts.
        for i in prange(n):
            # in-degrees (CSC: nonzeros in column i)
            for j in range(csc_indptr[i], csc_indptr[i + 1]):
                p = partition[csc_indices[j]]
                embeddings[i, p] += csc_data[j]
            # out-degrees (CSR: nonzeros in row i)
            for j in range(csr_indptr[i], csr_indptr[i + 1]):
                p = partition[csr_indices[j]]
                embeddings[i, k + p] += csr_data[j]

        return embeddings

    @staticmethod
    @njit
    def topk_indices(similarities, k):
        """Returns the indices of the top k elements in `similarities` in descending order.

        This helper function uses a simple partial selection sort.
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

    def compute_partition_from_frozen(
        self, frozen_indices, frozen_partition, similarity_matrix_to_frozen
    ):
        """Computes a partition for non-frozen points based on the top c closest majority frozen points."""

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
                top_c_indices = self.topk_indices(
                    similarities, self.topk
                )  # np.argsort(similarities)[-c:][::-1]  # Descending order

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

    def compute_reverse_mapping(self, csr_indptr, csr_indices, csr_data, n):
        """Compute the reverse mapping for a CSR matrix."""

        # First, count how many times each node appears as a neighbor.
        counts = np.zeros(n, dtype=np.int32)
        for i in range(n):
            for idx in range(csr_indptr[i], csr_indptr[i + 1]):
                node = csr_indices[idx]
                counts[node] += 1

        # Build the pointer array for the reverse mapping.
        rev_indptr = np.empty(n + 1, dtype=np.int32)
        rev_indptr[0] = 0
        for i in range(n):
            rev_indptr[i + 1] = rev_indptr[i] + counts[i]

        # Allocate arrays to hold the reverse mapping indices and the corresponding weights.
        rev_indices = np.empty(rev_indptr[-1], dtype=np.int32)
        rev_data = np.empty(rev_indptr[-1], dtype=csr_data.dtype)

        # Temporary counter array to keep track of insertion positions.
        temp = np.zeros(n, dtype=np.int32)
        for i in range(n):
            for idx in range(csr_indptr[i], csr_indptr[i + 1]):
                node = csr_indices[idx]
                pos = rev_indptr[node] + temp[node]
                rev_indices[pos] = i
                rev_data[pos] = csr_data[idx]
                temp[node] += 1

        return rev_indptr, rev_indices, rev_data

    @staticmethod
    @njit
    def incremental_embedding_update(
        embedding,
        pre_in_indptr,
        pre_in_indices,
        pre_in_data,
        pre_out_indptr,
        pre_out_indices,
        pre_out_data,
        old_partition,
        new_partition,
        k,
        changed_nodes,
    ):
        """Update the embedding matrix based on the changed nodes."""

        # For each node j that changed its partition:
        for j in changed_nodes:
            old_label = old_partition[j]
            new_label = new_partition[j]
            # Update all nodes i that have j as an in-neighbor.
            start = pre_in_indptr[j]
            end = pre_in_indptr[j + 1]
            for idx in range(start, end):
                i = pre_in_indices[idx]
                weight = pre_in_data[idx]  # Weight of edge i <- j.
                if new_label != -1:
                    embedding[i, new_label] += weight
                if old_label != -1:
                    embedding[i, old_label] -= weight

            # Update all nodes i that have j as an out-neighbor.
            start = pre_out_indptr[j]
            end = pre_out_indptr[j + 1]
            for idx in range(start, end):
                i = pre_out_indices[idx]
                weight = pre_out_data[
                    idx
                ]  # Weight of edge i <- j in the out-neighbor context.
                # Update the out-neighbor part (columns k to 2*k-1):
                if new_label != -1:
                    embedding[i, k + new_label] += weight
                if old_label != -1:
                    embedding[i, k + old_label] -= weight

    def fit(self, adj_csr, ground_truth_partition, tentative_labels=None):
        """Run NTAC on the given adjacency matrix and ground truth partition."""

        unlabeled_symbol = "?"
        # unlabeled_symbol = -1
        unique_labels = np.unique(ground_truth_partition)
        valid_labels = [lbl for lbl in unique_labels if lbl != unlabeled_symbol]

        # Create the mapping for valid labels starting at 0.
        label_mapping = {lbl: i for i, lbl in enumerate(valid_labels)}
        label_mapping[unlabeled_symbol] = -1
        k = len(valid_labels)
        reverse_mapping = {new: old for old, new in label_mapping.items()}
        ground_truth_partition = np.array(
            [
                label_mapping[p] if p != unlabeled_symbol else -1
                for p in ground_truth_partition
            ]
        )

        # We assume that tentative_labels is an array of strings (same length as ground_truth_partition)
        # and that a tentative label is any value other than unlabeled_symbol.
        if tentative_labels is not None:
            tentative_labels = np.array(
                [
                    label_mapping[p] if p != unlabeled_symbol else -1
                    for p in tentative_labels
                ]
            )

        time_spent_on_initial_embedding = 0.0
        time_spent_on_initial_sim = 0.0
        time_spent_on_embedding = 0.0
        time_spent_on_sim_matrix = 0.0

        # record the time spent on the whole algorithm
        t0 = time.time()
        adj_csc = adj_csr.tocsc()
        n = adj_csr.shape[0]
        labeled_nodes = np.where(ground_truth_partition != -1)[0]
        print(f"Percentage of labeled nodes: {len(labeled_nodes) / n * 100:.2f}%")
        _, counts = np.unique(ground_truth_partition[labeled_nodes], return_counts=True)
        num_singleton_clusters = np.sum(counts == 1)
        print(f"Number of clusters with only a single node: {num_singleton_clusters}")

        frac_labeled = self.ground_truth_frac  # fraction of labeled nodes to freeze
        # take random 10% of frac labelled nodes to be majority nodes
        frac_labeled = frac_labeled
        num_frozen = int(len(labeled_nodes) * frac_labeled)

        if self.force_all_clusters:
            # Ensure at least one node per cluster (only consider clusters that appear in labeled_nodes)
            unique_labels = np.unique(ground_truth_partition[labeled_nodes])
            frozen_set = set()
            for c in unique_labels:
                # Get the indices (in the full node space) of nodes with label c
                indices_c = labeled_nodes[
                    np.where(ground_truth_partition[labeled_nodes] == c)[0]
                ]
                frozen_set.add(np.random.choice(indices_c))
            remaining = num_frozen - len(frozen_set)
            if remaining > 0:
                possible_indices = np.setdiff1d(
                    labeled_nodes, np.array(list(frozen_set))
                )
                frozen_set.update(random.sample(list(possible_indices), remaining))
                assert len(frozen_set) == num_frozen
        else:
            frozen_set = set(random.sample(list(labeled_nodes), num_frozen))

        frozen_indices = np.array(list(frozen_set))

        # Define verification nodes as the remaining labeled nodes (do not include unlabeled nodes)
        verification_indices = np.setdiff1d(labeled_nodes, frozen_indices)
        # active_indices = np.setdiff1d(np.arange(n), frozen_indices)

        # add the tentative labels to the frozen indices
        if tentative_labels is not None:
            tentative_frozen_indices = np.where(tentative_labels != -1)[0]
            frozen_indices = np.concatenate((frozen_indices, tentative_frozen_indices))
            # check no frozen indices have label -1
            # a bit dangerous but should be fine
            ground_truth_partition = np.where(
                tentative_labels != -1, tentative_labels, ground_truth_partition
            )

        frozen_partition = ground_truth_partition[frozen_indices]
        frozen_percentage = len(frozen_indices) / n * 100
        if self.verbose:
            print(
                "num unique clusters in frozen: ",
                len(np.unique(frozen_partition)),
                "out of total clusters: ",
                k,
            )
            print(
                f"Percentage of frozen vertices: {frozen_percentage:.2f}%, Number of frozen vertices: {len(frozen_indices)}"
            )
        # print % labelled per location; #nodes with labels in location X / #total number of nodes in the location

        csr_indptr = adj_csr.indptr
        csr_indices = adj_csr.indices

        # Create the CSC representation and extract its arrays:
        adj_csc = adj_csr.tocsc()
        csc_indptr = adj_csc.indptr
        csc_indices = adj_csc.indices
        csc_data = adj_csc.data
        csr_data = adj_csr.data
        # Precompute reverse mappings for incremental updates:
        pre_out_indptr, pre_out_indices, pre_out_data = self.compute_reverse_mapping(
            csr_indptr, csr_indices, csr_data, n
        )
        pre_in_indptr, pre_in_indices, pre_in_data = self.compute_reverse_mapping(
            csc_indptr, csc_indices, csc_data, n
        )
        _t = time.time()

        initial_embedding = self.generate_initial_embedding(
            csr_indptr, csr_indices, csc_indptr, csc_indices
        )

        time_spent_on_initial_embedding += time.time() - _t

        _t = time.time()

        initial_similarity_matrix_to_frozen = self.generate_similarity_matrix_to_frozen(
            initial_embedding, frozen_indices
        )
        time_spent_on_initial_sim += time.time() - _t

        partition = self.compute_partition_from_frozen(
            frozen_indices, frozen_partition, initial_similarity_matrix_to_frozen
        )

        _t = time.time()
        embedding = self.generate_embedding(
            csr_indptr,
            csr_indices,
            csr_data,
            csc_indptr,
            csc_indices,
            csc_data,
            partition,
            k,
        )
        time_spent_on_embedding += time.time() - _t
        _t = time.time()
        similarity_matrix_to_frozen = self.generate_similarity_matrix_to_frozen(
            embedding, frozen_indices
        )

        time_spent_on_sim_matrix += time.time() - _t

        pbar = tqdm(range(self.rounds))

        time_spent_on_partition = 0.0

        def print_times():
            print(
                f"Time spent on initial embedding: {time_spent_on_initial_embedding:.2f} seconds"
            )
            print(
                f"Time spent on initial similarity matrix: {time_spent_on_initial_sim:.2f} seconds"
            )
            print(f"Time spent on embedding: {time_spent_on_embedding:.2f} seconds")
            print(
                f"Time spent on similarity matrix update: {time_spent_on_sim_matrix:.2f} seconds"
            )
            print(f"Time spent on partition: {time_spent_on_partition:.2f} seconds")

        last_embedding = embedding.copy()
        for round in pbar:
            _t = time.time()
            old_partition = partition.copy()
            partition = self.compute_partition_from_frozen(
                frozen_indices, frozen_partition, similarity_matrix_to_frozen
            )

            changed_nodes = np.nonzero(partition != old_partition)[0]
            # print("Percentage of nodes that changed partition: ", changed_nodes.size / len(partition))

            time_spent_on_partition += time.time() - _t

            _t = time.time()

            if changed_nodes.size > 0:
                self.incremental_embedding_update(
                    last_embedding,
                    pre_in_indptr,
                    pre_in_indices,
                    pre_in_data,
                    pre_out_indptr,
                    pre_out_indices,
                    pre_out_data,
                    old_partition,
                    partition,
                    k,
                    changed_nodes,
                )
            embedding = (1 - self.lr) * embedding + self.lr * last_embedding

            time_spent_on_embedding += time.time() - _t

            _t = time.time()
            similarity_matrix_to_frozen = self.generate_similarity_matrix_to_frozen(
                embedding, frozen_indices
            )
            time_spent_on_sim_matrix += time.time() - _t

            # update_metrics()
            # pbar.set_description(f'Round {round} ARI excluding gt: {metrics["ari"][-1]}, acc: {metrics["acc"][-1]}, f1: {metrics["f1"][-1]}')
            # assigned_indices = [idx for idx in verification_indices if partition[idx] != -1]
            # if self.verbose:
            pbar.set_description(
                f"Round {round}"
            )  # , acc: {metrics["acc"][-1]}, f1: {metrics["f1"][-1]}')
            # print("Region-wise accuracies (final round):")
            # final_region_acc = metrics['detailed_acc'][-1]
            # for region, acc in final_region_acc.items():
            #     print(f"  {region}: {acc}")
            if self.verbose:
                print_times()

        total_time = time.time() - t0
        if self.verbose:
            print(f"Total time: {total_time:.2f} seconds")
            print_times()
        # map to original labels
        partition = np.array([reverse_mapping[p] for p in partition])
        partition[verification_indices] = ground_truth_partition[verification_indices]
        return partition, frozen_indices, embedding, label_mapping, verification_indices

        # return partition, metrics, frozen_indices, embedding, label_mapping, verification_indices
