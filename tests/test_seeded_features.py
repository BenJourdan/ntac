from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from ntac import GraphData, Ntac


def test_seeded_ntac_uses_external_node_features_for_assignment() -> None:
    adjacency = csr_matrix((3, 3), dtype=float)
    labels = np.array(["A", "B", "?"], dtype=object)
    node_features = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.9, 0.1],
        ],
        dtype=float,
    )

    model = Ntac(
        data=adjacency,
        labels=labels,
        node_features=node_features,
        feature_weight=1.0,
    )
    model.step()

    partition = model.get_partition()
    assert partition.tolist() == ["A", "B", "A"]


def test_graph_data_stores_optional_node_features() -> None:
    adjacency = csr_matrix((2, 2), dtype=float)
    labels = np.array(["A", "?"], dtype=object)
    node_features = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)

    data = GraphData(adjacency, labels=labels, node_features=node_features)

    assert data.node_features is not None
    assert data.node_features.shape == (2, 2)
    assert np.allclose(data.node_features, node_features)


def test_seeded_ntac_rejects_negative_node_features() -> None:
    adjacency = csr_matrix((2, 2), dtype=float)
    labels = np.array(["A", "?"], dtype=object)
    node_features = np.array([[1.0], [-1.0]], dtype=float)

    with pytest.raises(ValueError, match="node_features must be nonnegative"):
        Ntac(
            data=adjacency,
            labels=labels,
            node_features=node_features,
            feature_weight=1.0,
        )
