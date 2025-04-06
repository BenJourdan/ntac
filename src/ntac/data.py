"""Download and unzip the FlyWire data if not already cached."""

import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score, adjusted_rand_score, f1_score

DATA_URL = (
    "https://github.com/BenJourdan/ntac/releases/download/v0.1.0/dynamic_data.zip"
)


class GraphData:
    """Class to load and process graph data for the NTAC package."""

    def __init__(
        self,
        edges_file,
        types_file,
        ignore_side=False,
        target_locations=None,
        return_neuron_ids=False,
        top_regions_summary_file=None,
    ):
        """Initialize the GraphData object."""
        self.edges_file = edges_file
        self.types_file = types_file
        self.ignore_side = ignore_side
        self.target_locations = target_locations
        self.return_neuron_ids = return_neuron_ids
        self.top_regions_summary_file = top_regions_summary_file
        self.load_data()

    def load_data(self):
        """Load the graph data and partition information."""
        results = self.load_graph_and_partition()
        if self.return_neuron_ids:
            (
                self.adj_csr,
                self.ground_truth_partition,
                self.idx_to_node,
                self.features,
                self.locations,
                self.node_top_regions,
                self.n,
                self.top_regions_summary,
                self.cluster_capacities,
                self.node_to_neuron_id,
            ) = results
        else:
            (
                self.adj_csr,
                self.ground_truth_partition,
                self.idx_to_node,
                self.features,
                self.locations,
                self.node_top_regions,
                self.n,
                self.top_regions_summary,
                self.cluster_capacities,
            ) = results

    def get_metrics(self, partition, indices=None):
        """Compute metrics for the given partition."""
        metrics = {}
        if indices is None:
            indices = np.arange(self.n)
        assigned_indices = [idx for idx in indices if partition[idx] != -1]
        metrics["ari"] = adjusted_rand_score(
            self.ground_truth_partition[assigned_indices], partition[assigned_indices]
        )
        metrics["f1"] = f1_score(
            self.ground_truth_partition[assigned_indices],
            partition[assigned_indices],
            average="macro",
        )
        metrics["acc"] = accuracy_score(
            self.ground_truth_partition[assigned_indices], partition[assigned_indices]
        )

        if self.locations is not None:
            location_labels = {}
            for i in range(self.n):
                location = self.locations[i]
                if location not in location_labels:
                    location_labels[location] = []
                location_labels[location].append(i)
            for location, indices in location_labels.items():
                num_labeled = len(np.intersect1d(indices, assigned_indices))
                num_total = len(indices)
                print(
                    f"Location {location}: {num_labeled / num_total * 100:.2f}% labeled ({num_labeled} / {num_total})"
                )
        region_groups = {}
        for i in assigned_indices:
            region = self.locations[i] if self.locations is not None else "unknown"
            region_groups.setdefault(region, []).append(i)

        region_acc = {}
        for region, indices in region_groups.items():
            indices = np.intersect1d(indices, assigned_indices)
            if len(indices) == 0:
                region_acc[region] = -1
                continue
            region_acc[region] = accuracy_score(
                self.ground_truth_partition[indices], partition[indices]
            )
        metrics["region_acc"] = region_acc

        return metrics

    def load_graph_and_partition(self):
        """Load the graph and partition data from the specified files."""
        # first check if we have downloaded and unzipped the data
        GraphData.download_flywire_data()

        # === 1. Load all data without filtering ===
        # Load edges and types files
        edges_df = pd.read_csv(self.edges_file)
        types_df = pd.read_csv(self.types_file)

        # Get the full set of nodes from the edges file (do not filter yet)
        nodes = set(pd.unique(edges_df[["from", "to"]].values.ravel("K")))

        # Build a mapping from node name to index (sorting ensures deterministic ordering).
        node_to_idx = {node: idx for idx, node in enumerate(sorted(nodes))}
        idx_to_node = {idx: node for node, idx in node_to_idx.items()}
        n = len(nodes)

        # Build the initial adjacency matrix using all nodes.
        row = edges_df["from"].map(node_to_idx).values
        col = edges_df["to"].map(node_to_idx).values
        data = edges_df["weight"].values
        adj_csr = csr_matrix((data, (row, col)), shape=(n, n))

        # Initialize arrays for ground truth labels, features, and locations.
        ground_truth_partition = np.full(n, "?", dtype=object)
        features = np.zeros((n, 3), dtype=np.float32)  # assume 3 features per node
        locations = [None] * n

        # Initialize top_regions if the new column exists.
        if "top in/out region(s)" in types_df.columns:
            top_regions = [None] * n
        else:
            top_regions = None

        # Populate arrays from the types dataframe.
        for _, row_data in types_df.iterrows():
            vertex = row_data["vertex"]
            if vertex in node_to_idx:
                idx = node_to_idx[vertex]
                # Set ground truth partition (if cluster info is missing, keep as '?')
                try:
                    if pd.isna(row_data["cluster"]):
                        ground_truth_partition[idx] = "?"
                    else:
                        ground_truth_partition[idx] = row_data["cluster"]
                except Exception:
                    ground_truth_partition[idx] = "?"

                # Extract features (e.g., cable_length, surface_area, volume).
                row_features = [
                    row_data["cable_length"],
                    row_data["surface_area"],
                    row_data["volume"],
                ]
                if row_features == ["?", "?", "?"]:
                    features[idx] = [0, 0, 0]
                else:
                    features[idx] = row_features

                # Build the location string (optionally including side).
                if "side" in row_data and "compartment" in row_data:
                    loc = row_data["compartment"]
                    if not self.ignore_side:
                        loc = f"{row_data['side']} {loc}"
                    locations[idx] = loc
                    # print(f"Location for {vertex}: {loc}")

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
            idx_to_node = {
                new_idx: idx_to_node[old_idx]
                for new_idx, old_idx in enumerate(valid_indices)
            }
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
                fw_type = row_data["FlyWire visual type"]
                expected_cells = row_data["Expected number of cells on each side"]
                most_common_top = row_data["Most common top in/out region(s)"]
                percentage = row_data[
                    "Percentage of cells of this type and these top i/o regions"
                ]
                # Convert percentage to float
                percentage = float(percentage.strip("%")) / 100
                top_regions_summary[fw_type] = (
                    expected_cells,
                    most_common_top,
                    percentage,
                )
                cluster_capacities[fw_type] = expected_cells

        # === 4. Return the results ===
        # (You can later use `top_regions` and `cluster_capacities` to filter neurons whose top region
        # does not match the expected capacity and region for their cluster.)
        if not self.return_neuron_ids:
            return (
                adj_csr,
                ground_truth_partition,
                idx_to_node,
                features,
                locations,
                top_regions,
                n,
                top_regions_summary,
                cluster_capacities,
            )
        else:
            # Extract the neuron id for each vertex from the types file.
            node_to_neuron_id = {}
            for _, row_data in types_df.iterrows():
                vertex = row_data["vertex"]
                neuron_id = row_data["neuron id"]
                if vertex in node_to_idx:
                    node_to_neuron_id[vertex] = neuron_id
            return (
                adj_csr,
                ground_truth_partition,
                idx_to_node,
                features,
                locations,
                top_regions,
                n,
                top_regions_summary,
                cluster_capacities,
                node_to_neuron_id,
            )

    @staticmethod
    def download_flywire_data(
        data_url=DATA_URL,
        cache_dir=os.path.join(os.path.expanduser("~"), ".ntac"),
        zip_path=None,
        data_dir=None,
        verbose=False,
    ):
        """Download and unzip the FlyWire data if not already cached.

        Parameters
        ----------
        data_url : str
            URL to download the FlyWire data from.
        cache_dir : str
            Directory to cache the downloaded data.
        zip_path : str
            Path to the zip file.
        data_dir : str
            Directory to extract the data to.
        verbose : bool
            If True, print progress messages.

        """
        if zip_path is None:
            zip_path = os.path.join(cache_dir, "dynamic_data.zip")
        if data_dir is None:
            data_dir = os.path.join(cache_dir, "flywire_data")

        os.makedirs(cache_dir, exist_ok=True)

        # Step 1: Download if missing
        if not os.path.exists(zip_path):
            if verbose:
                print("Downloading FlyWire data...")
            urllib.request.urlretrieve(data_url, zip_path)
            if verbose:
                print("Download complete.")

        # Step 2: Unzip if not already done
        if not os.path.exists(data_dir):
            if verbose:
                print("Unzipping FlyWire data...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(data_dir)
            if verbose:
                print("Unzip complete.")

        if verbose:
            print(f"FlyWire data available at: {data_dir}")
        return data_dir
