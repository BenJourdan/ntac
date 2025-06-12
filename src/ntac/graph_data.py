
import pandas as pd
from sklearn.metrics import adjusted_rand_score, accuracy_score, f1_score
from scipy.sparse import csr_matrix
import numpy as np
import random

class GraphData:
    """
    Class for storing graph data and computing evaluation metrics.
    
    Attributes:
        adj_csr (scipy.sparse.csr_matrix): The adjacency matrix in CSR format.
        adj_csc (scipy.sparse.csc_matrix): The adjacency matrix converted to CSC format.
        labels (np.array): Array of labels (ground truth partitions) for each node.
        n (int): The number of nodes in the graph.
        unlabeled_symbol (str): Symbol used for nodes that have no label.
        labeled_nodes (np.array): Indices of nodes that are labeled.
        unique_labels (np.array): Unique set of labels found in the graph.
    """
    def __init__(self, adj_csr, labels):
        """
        Initialize GraphData with the provided adjacency matrix and labels.
        
        Parameters:
            adj_csr (scipy.sparse.csr_matrix): Sparse adjacency matrix in CSR format.
            labels (np.array): Array of labels (one per node); unlabeled nodes have label "?".
        """
        self.adj_csr = adj_csr
        self.adj_csc = adj_csr.tocsc()
        self.labels = labels.copy()
        self.n = adj_csr.shape[0]
        self.unlabeled_symbol = "?"
        self.labeled_nodes = np.where(self.labels != self.unlabeled_symbol)[0]
        #self.unique_labels = np.unique(self.labels)
        self.unique_labels  = np.unique(self.labels[self.labeled_nodes])

 
    def get_metrics(self, partition, indices, gt_labels, map_labels=False):
        #TODO: make a pass and fix the doc strings at the end
        """
        Calculate evaluation metrics over the specified node indices.

        Computes the Adjusted Rand Index (ARI), macro F1 score, and accuracy
        comparing the given partition against the ground truth labels.

        Parameters
        ----------
        partition : np.array
            Array of predicted labels for each node.
        indices : np.array
            Indices of the nodes to evaluate.
        gt_labels : np.array
            Array of ground truth labels for nodes.

        Returns
        -------
        dict
            A dictionary with metric names as keys and the corresponding scores:

            - `'ari'`: Adjusted Rand Index
            - `'f1'`: macro F1 score
            - `'acc'`: classification accuracy
        """
        metrics = {}
        if not isinstance(partition, dict):
            # If partition is not a dict, convert it to a dict with indices as keys
            partition = {i: [(label, 1.0)] for i, label in enumerate(partition)}
        top_k_partition = partition
        #turn partition into a 1-D array by taking only the first label
        partition = np.empty(self.n, dtype=object)
        for i in range(self.n):
            partition[i] = top_k_partition[i][0][0]

        some_node = next(iter(top_k_partition))
        K = len(top_k_partition[some_node])

        topk_acc = {}
        n_test = len(indices)

        # for each k=1…K, count how many test‐nodes have the true label in their top‐k
        for k in range(1, K+1):
            correct = 0
            for i in indices:
                # get the first k predicted labels for node i
                preds_i = [label for (label, _) in top_k_partition[i][:k]]
                if gt_labels[i] in preds_i:
                    correct += 1
            topk_acc[k] = correct / n_test
        metrics['topk_acc'] = topk_acc

        #turn partition into a 1-D array by taking only the first label
        partition = np.array([top_k_partition[i][0][0] for i in range(self.n)])
    

        metrics["ari"] = adjusted_rand_score(gt_labels[indices], partition[indices])
        metrics["f1"] = f1_score(gt_labels[indices], partition[indices], average='weighted')
        metrics["acc"] = accuracy_score(gt_labels[indices], partition[indices])

        return metrics

 
    def test_train_split(self, train_size=0.1, sampling_type="at_least_one_per_class",  random_seed=None):
        """
        Split the labeled nodes into training and test sets using various sampling strategies.

        Parameters
        ----------
        train_size : float or int
            If < 1, the fraction of total labeled nodes to use as training;
            if ≥ 1, the absolute number of training samples.
        sampling_type : str
            Type of sampling to use. One of:
            - `"uniform"`: Random sampling from all labeled nodes.
            - `"at_least_one_per_class"`: Ensure at least one sample per class,
            then randomly fill the remaining slots. This may exceed `train_size`
            if needed to satisfy the class constraint.
            - `"exactly_k_per_class"`: Sample exactly `train_size` nodes per class.
            - `"stratified"`: Sample nodes so that the class distribution in the
            training set mirrors the overall distribution.

        num_per_class : int, optional
            Number of samples per class, used only in `"exactly_k_per_class"` mode.

        Returns
        -------
        train_set : np.ndarray
            Array of indices for training nodes.
        test_set : np.ndarray
            Array of indices for test nodes.
        """
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        total_nodes = len(self.labeled_nodes)
        # Determine number of training samples
        n_train = int(total_nodes * train_size) if train_size <= 1 else train_size
        
        labeled_nodes = self.labeled_nodes  # Assumed to be a list or numpy array of indices.
        train_set = set()
        if sampling_type == "uniform":
            # Uniform random sampling from all labeled nodes.
            train_set = set(random.sample(list(labeled_nodes), n_train))
        
        elif sampling_type == "at_least_one_per_class":
            # Ensure at least one sample from each class found in labeled_nodes.
            
            for c in self.unique_labels:
                # Find indices (from the full node space) for nodes with label c that are in labeled_nodes.
                indices_c = labeled_nodes[np.where(self.labels[labeled_nodes] == c)[0]]
                assert len(indices_c) > 0, f"Class {c} has no labeled nodes."
                    
                #     # either warn or raise a more informative error:
                #     print(f"Warning: no labeled nodes for class {c}, skipping.")
                #     continue
                # Choose one sample from this class.
                train_set.add(np.random.choice(indices_c))
            
            # Fill the rest uniformly.
            remaining = n_train - len(train_set)
            if remaining > 0:
                possible_indices = np.setdiff1d(labeled_nodes, np.array(list(train_set)))
                if len(possible_indices) < remaining:
                    raise ValueError("Not enough nodes to complete the training set")
                train_set.update(random.sample(list(possible_indices), remaining))
        
        elif sampling_type == "exactly_k_per_class":
            #Make sure train_size is an integer
            if not isinstance(train_size, int):
                raise ValueError("train_size must be an integer for exactly_k_per_class sampling")
            for c in self.unique_labels:
                indices_c = labeled_nodes[np.where(self.labels[labeled_nodes] == c)[0]]
                if len(indices_c) < train_size:
                    #take the entire class and print warning
                    print(f"Warning: class {c}: needed {train_size} but found {len(indices_c)} elements. Taking all.")
                   
                chosen = random.sample(list(indices_c), min(train_size, len(indices_c)))
                train_set.update(chosen)
            
        
        elif sampling_type == "stratified":
            # Stratified sampling to mimic the overall class distribution in the training set.
            
            # Get labels for all labeled_nodes.
            labels_array = self.labels[labeled_nodes]
            # Sample per class based on proportion.
            for c in self.unique_labels:
                indices_c = labeled_nodes[np.where(labels_array == c)[0]]
                # Calculate the count for this class, rounding as necessary.
                count = int(round((len(indices_c) / total_nodes) * n_train))
                # Ensure at least one sample is taken from a class that appears.
                if count == 0 and len(indices_c) > 0:
                    count = 1
                if count > len(indices_c):
                    print(f"Warning: class {c}: needed {count} but found {len(indices_c)} elements. Taking all.")
                chosen = random.sample(list(indices_c), count)
                train_set.update(chosen)
            
            # Adjust total count if rounding left us with too few or too many samples.
            if len(train_set) < n_train:
                remaining = n_train - len(train_set)
                possible_indices = np.setdiff1d(labeled_nodes, np.array(list(train_set)))
                if len(possible_indices) < remaining:
                    raise ValueError("Not enough nodes to complete the stratified training set")
                train_set.update(random.sample(list(possible_indices), remaining))
            elif len(train_set) > n_train:
                train_set = set(random.sample(list(train_set), n_train))
        
        else:
            raise ValueError(f"Unknown sampling type: {sampling_type}")
        
        # Define the test set as the remainder of labeled_nodes.
        test_set = set(labeled_nodes) - train_set

        # Return as numpy arrays.
        return np.array(list(train_set)), np.array(list(test_set))


class FAFBData(GraphData):
    """
    Class for loading FAFB dataset data, extending GraphData with additional attributes.
    
    Loads graph data (edges), node types, and optionally a top regions summary from CSV files.
    Provides data filtering (e.g., by target locations) and extracts extra features such as 
    node locations and top regions.
    
    Attributes:
        edges_file (str): Path to the CSV file containing edge data.
        types_file (str): Path to the CSV file containing node type/label data.
        ignore_side (bool): Flag to determine whether to ignore the 'side' information when processing locations.
        target_locations (list or set, optional): Specific locations to filter the nodes.
        top_regions_summary_file (str, optional): Path to the CSV file for top regions summary.
        (Additional attributes such as adjacency matrices, ground truth partitions,
         features, and mappings are loaded by load_graph_and_partition().)
    """
    def __init__(self, edges_file, types_file, ignore_side=False, target_locations=None, top_regions_summary_file=None):
        """
        Initialize FAFBData by loading graph data and node partition information from provided files.
        
        Parameters:
            edges_file (str): Path to the CSV file containing edges (with 'from', 'to', and 'weight' columns).
            types_file (str): Path to the CSV file containing node type/label and feature data.
            ignore_side (bool): Whether to ignore side information when forming location strings.
            target_locations (list, optional): If provided, filter nodes to include only those with locations in this list.
            top_regions_summary_file (str, optional): Path to a CSV file summarizing top regions data.
        """
        self.edges_file = edges_file
        self.types_file = types_file
        self.ignore_side = ignore_side #TODO: should be removed after the datasets will be in the same format
        self.target_locations = target_locations
        self.top_regions_summary_file = top_regions_summary_file
        self.unlabeled_symbol = "?"
        self.load_graph_and_partition()
        # (self.adj_csr, self.ground_truth_partition, self.idx_to_node, self.features,
        #      self.locations, self.node_top_regions, self.n, self.top_regions_summary,
        #      self.cluster_capacities) = self.load_graph_and_partition()
        #call parent constructor
        super().__init__(self.adj_csr, self.ground_truth_partition)

    

    def get_metrics(self, partition, indices, gt_labels, compute_class_acc=False):
        """
        Compute overall, top-k, region-level, and class-level accuracy metrics for FAFB data.

        This function accepts either:
        - A “hard” partition as a 1-D array of length n (one label per node), or
        - A dictionary of the form `{node_index: [(label₁, score₁), (label₂, score₂), …]}`,
        as returned by `nt.get_topk_partition(K)`.

        Parameters
        ----------
        partition : np.ndarray or dict
            If an array of shape (n,), each entry is a single predicted label.
            If a dict mapping node_index to a list of (label, score) pairs, this is interpreted
            as a top-k ranking for each node.
        indices : array-like of int
            Subset of node indices over which to evaluate metrics (e.g., test set indices).
        gt_labels : np.ndarray of shape (n,)
            Ground-truth labels for all nodes.
        compute_class_acc : bool, optional
            If True, compute class-level top-k accuracy for each label in `self.unique_labels`.
            Default is False.

        Returns
        -------
        dict
            A dictionary containing:

            - `'topk_acc'`: dict mapping k → overall accuracy@k over `indices`.
            - `'topk_region_acc'`: dict mapping region → list of [acc@1, acc@2, ..., acc@K] (if locations exist).
            - `'region_acc'`: dict mapping region → accuracy@1 (derived from `topk_region_acc`).
            - `'topk_class_acc'`: dict mapping class_label → [acc@1, ..., acc@K] (if `compute_class_acc=True`).
            - `'class_acc'`: dict mapping class_label → accuracy@1 (if `compute_class_acc=True`).
            - Any additional keys returned by `super().get_metrics(...)`, such as `'acc'`, `'ari'`, or `'f1'`.
        """

        metrics = super().get_metrics(partition, indices, gt_labels)
        if not isinstance(partition, dict):
            # If partition is not a dict, convert it to a dict with indices as keys
            partition = {i: [(label, 1.0)] for i, label in enumerate(partition)}
        top_k_partition = partition
        #turn partition into a 1-D array by taking only the first label
        partition = np.empty(self.n, dtype=object)
        for i in range(self.n):
            partition[i] = top_k_partition[i][0][0]

        some_node = next(iter(top_k_partition))
        K = len(top_k_partition[some_node])


        #compute top_k_region_acc
        if self.locations is not None and not all(loc is None for loc in self.locations):
            region_groups = {}
            for i in indices:
                region = self.locations[i] if self.locations is not None else "unknown"
                region_groups.setdefault(region, []).append(i)

            top_k_region_acc = {}
            for region, reg_indices in region_groups.items():
                top_k_region_acc[region] = []
                for k in range(1, K+1):
                    correct = 0
                    for i in reg_indices:
                        preds_i = [pred[0] for pred in top_k_partition[i][:k]]
                        if gt_labels[i] in preds_i:
                            correct += 1
                    if len(reg_indices) == 0:
                        top_k_region_acc[region].append(np.nan)
                    else:
                        top_k_region_acc[region].append(correct / len(reg_indices))
                    
            metrics['topk_region_acc'] = top_k_region_acc
            metrics['region_acc'] = {k: v[0] for k, v in metrics['topk_region_acc'].items()}

        #compute top_k_class_acc (similar to region_acc)
        if compute_class_acc:
            class_acc = {}
            for label in self.unique_labels:
                class_acc[label] = []
                for k in range(1, K+1):
                    correct = 0
                    for i in indices:
                        preds_i = [pred[0] for pred in top_k_partition[i][:k]]
                        if gt_labels[i] == label and gt_labels[i] in preds_i:
                            correct += 1
                    if len(indices) == 0:
                        print(f"Warning: no indices for class {label} in top_k_class_acc")
                        class_acc[label].append(np.nan)
                    else:
                        class_acc[label].append(correct / len(indices))
            metrics['topk_class_acc'] = class_acc
            metrics['class_acc'] = {k: v[0] for k, v in metrics['topk_class_acc'].items()}
        

        return metrics


    #TODO: make the file format for BANC and FAFB the same and change accordingly
    def load_graph_and_partition(self):
        """
        Loads graph structure and node data from CSV files; processes and filters data as required.
        
        This method performs the following:
         1. Loads edges and types from the provided CSV files.
         2. Builds the complete node set and a mapping from node names to indices.
         3. Constructs the initial adjacency matrix (CSR format).
         4. Initializes ground truth labels, features, and locations from the types file.
         5. Optionally filters nodes based on target locations if specified.
         6. Optionally loads a top regions summary file to build capacity mappings.
        
        Returns:
            tuple: Contains the following elements in order:
                - adj_csr (scipy.sparse.csr_matrix): Adjacency matrix for the graph.
                - ground_truth_partition (np.array): Array of node labels (using self.unlabeled_symbol for missing labels).
                - idx_to_node (dict): Mapping from new node indices to original node names.
                - features (np.array): Node features array (assumed to have 3 features per node).
                - locations (list): List of location strings for each node.
                - top_regions (list or None): List of top in/out region(s) for each node, if available.
                - n (int): Number of nodes after filtering.
                - top_regions_summary (dict or None): Summary data for top regions, if provided.
                - cluster_capacities (dict or None): Mapping of cluster types to their expected capacities.
        """
        # === 1. Load all data without filtering ===
        edges_df = pd.read_csv(self.edges_file)
        types_df = pd.read_csv(self.types_file)
    
        # Get the full set of nodes from the edges file (do not filter yet)
        nodes = set(pd.unique(edges_df[['from', 'to']].values.ravel('K')))
    
        # Build a mapping from node name to index (sorting ensures deterministic ordering).
        node_to_idx = {node: idx for idx, node in enumerate(sorted(nodes))}
        idx_to_node = {idx: node for node, idx in node_to_idx.items()}
        n = len(nodes)
    
        # Build the initial adjacency matrix using all nodes.
        row = edges_df['from'].map(node_to_idx).values
        col = edges_df['to'].map(node_to_idx).values
        data = edges_df["weight"].values
        adj_csr = csr_matrix((data, (row, col)), shape=(n, n))
    
        # Initialize arrays for ground truth labels, features, and locations.
        ground_truth_partition = np.full(n, self.unlabeled_symbol, dtype=object)
        features = np.zeros((n, 3), dtype=np.float32)  # assume 3 features per node
        locations = [None] * n
    
        # Initialize top_regions if the new column exists.
        if "top in/out region(s)" in types_df.columns:
            top_regions = [None] * n
        else:
            top_regions = None
    
        # Populate arrays from the types dataframe.
        for _, row_data in types_df.iterrows():
            vertex = row_data['vertex']
            if vertex in node_to_idx:
                idx = node_to_idx[vertex]
                # Set ground truth partition (if cluster info is missing, keep as '?')
                try:
                    if pd.isna(row_data['cluster']):
                        ground_truth_partition[idx] = self.unlabeled_symbol
                    else:
                        ground_truth_partition[idx] = row_data['cluster']
                except Exception:
                    print(f"Error processing cluster for vertex {vertex}: {row_data['cluster']}")   
                    ground_truth_partition[idx] = self.unlabeled_symbol
    
                # Extract features (e.g., cable_length, surface_area, volume).
                row_features = [row_data['cable_length'], row_data['surface_area'], row_data['volume']]
                if row_features == ['?', '?', '?']:
                    features[idx] = [0, 0, 0]
                else:
                    features[idx] = row_features
            
                # Build the location string (optionally including side).
                if 'side' in row_data and 'compartment' in row_data:
                    loc = row_data['compartment']
                    if not self.ignore_side:
                        loc = f"{row_data['side']} {loc}"
                    locations[idx] = loc
            
                # If available, store the top in/out region for this neuron.
                if top_regions is not None:
                    top_regions[idx] = row_data["top in/out region(s)"]
    
        # === 2. Filter at the very end (if target_locations is provided) ===
        if self.target_locations is not None:
            # Build a list of valid indices: include a node only if its location exists and is in target_locations.
            valid_indices = []
            for i, loc in enumerate(locations):
                if loc is not None and loc in self.target_locations:
                    valid_indices.append(i)
            valid_indices = np.array(valid_indices)

            # Rebuild the adjacency matrix, ground truth labels, features, locations, and top_regions using only the valid nodes.
            adj_csr = adj_csr[valid_indices, :][:, valid_indices]
            ground_truth_partition = ground_truth_partition[valid_indices]
            features = features[valid_indices]
            locations = [locations[i] for i in valid_indices]
            if top_regions is not None:
                top_regions = [top_regions[i] for i in valid_indices]
    
            # Rebuild the idx_to_node mapping so that indices run from 0 to (n_filtered - 1)
            idx_to_node = {new_idx: idx_to_node[old_idx] for new_idx, old_idx in enumerate(valid_indices)}
            n = len(valid_indices)
    
        # === 3. Load the top regions summary file and create cluster capacities dict (if provided) ===
        top_regions_summary = None
        cluster_capacities = None
        if self.top_regions_summary_file is not None:
            top_regions_summary_df = pd.read_csv(self.top_regions_summary_file)
            top_regions_summary = {}
            cluster_capacities = {}
            # Expected columns:
            # 'FlyWire visual type', 'Expected number of cells on each side',
            # 'Most common top in/out region(s)', 'Percentage of cells of this type and these top i/o regions'
            for _, row_data in top_regions_summary_df.iterrows():
                fw_type = row_data['FlyWire visual type']
                expected_cells = row_data['Expected number of cells on each side']
                most_common_top = row_data['Most common top in/out region(s)']
                percentage = row_data['Percentage of cells of this type and these top i/o regions']
                # Convert percentage to float
                percentage = float(percentage.strip('%')) / 100
                top_regions_summary[fw_type] = (expected_cells, most_common_top, percentage)
                cluster_capacities[fw_type] = expected_cells
    
        # === 4. Return the results ===
        # (You can later use `top_regions` and `cluster_capacities` to filter neurons whose top region
        # does not match the expected capacity and region for their cluster.)
        node_to_neuron_id = {}
        for _, row_data in types_df.iterrows():
            vertex = row_data["vertex"]
            neuron_id = row_data["neuron id"]
            if vertex in node_to_idx:
                node_to_neuron_id[vertex] = neuron_id

        #check if contains unlabeled nodes and print warning
        if self.unlabeled_symbol in ground_truth_partition:
            print(f"Warning: {self.unlabeled_symbol} found in ground truth partition. This may affect evaluation metrics.")
            #print how many times and where
            counts = np.bincount(ground_truth_partition == self.unlabeled_symbol)
            print(f"Counts of {self.unlabeled_symbol}: {counts}")

        self.node_to_neuron_id = node_to_neuron_id
        self.adj_csr = adj_csr
        self.ground_truth_partition = ground_truth_partition
        self.idx_to_node = idx_to_node
        self.features = features
        self.locations = locations
        self.node_top_regions = top_regions
        self.n = n
        self.top_regions_summary = top_regions_summary
        self.cluster_capacities = cluster_capacities
        # return (adj_csr, ground_truth_partition, idx_to_node, features, locations,
        #             top_regions, n, top_regions_summary, cluster_capacities)
        
    def get_neuron_ids(self, indices):
        """
        Retrieve the neuron IDs for a given set of node indices.
        
        Parameters:
            indices (np.array): Array of node indices for which to retrieve neuron IDs.
        
        Returns:
            list: List of neuron IDs corresponding to the provided indices.
        """
        return [self.node_to_neuron_id[self.idx_to_node[i]] for i in indices]