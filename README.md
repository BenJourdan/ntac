# NTAC  <img src="assets//logo.png" align="right" height="100" />

<div style="margin-top: 30px;"></div>

<br>

This is a Python implementation of the Neuronal typing algorithm described in

[Gregory Schwartzman, Ben Jourdan, David García-Soriano, Arie Matsliah. Connectivity Is All You Need: Inferring Neuronal Types with NTAC. bioRxiv. 2025](https://www.biorxiv.org/cgi/content/short/2025.06.11.659184v1)

## Neuronal Type Assignment from Connectivity

NTAC (Neuronal Type Assignment from Connectivity) groups neurons into cell types based solely on synaptic connectivity. It comes in two variants:

- **Seeded (semi-supervised):** Requires a small fraction of neurons with known labels.  
- **Unseeded (unsupervised):** Requires no labels.

## Installation:

Install NTAC with:
`pip install ntac`


Install cudatoolkit with conda for optional speed for unseeded:
`conda install -c conda-forge cudatoolkit`



## Qickstart with NTAC:
```python
import numpy as np
import scipy.sparse as sp
from ntac import Ntac, sbm, GraphData

# Generate an adjacency matrix and labels from an SBM graph 
A, labels = sbm(n=1000, k=4)



#NTAC requires a CSR array and labels as a string array
A_csr = sp.csr_array(A)
labels = np.array([str(l) for l in labels])
# NTAC can take in as input a CSR matrix and labels, 
# but it is easier to use the GraphData class for test_train split and metrics
data = GraphData(A_csr, labels=labels)

############################################
#Seeded NTAC
#use only 10% for training
train_indices, test_indices = data.test_train_split(train_size=0.1)
labels[test_indices] = "?" # "?" indicates a nodes is unlabeled
#Initialize NTAC with the data and labels
nt = Ntac(data=data, labels=labels)
# nt = Ntac(data=A_csr, labels=labels) # if you want to use CSR matrix directly
for i in range(5):
    print(f"Step {i}")
    nt.step()
    partition = nt.get_partition()
    #partition = nt.get_topk_partition(5) # if we want to get the top 5 labels for each node
    metrics = data.get_metrics(partition, test_indices, data.labels)
    print(f"Accuracy: {metrics['acc']:.3f}") #can also get ARI, weighted F1, and topk accuracy (if using get_topk_partition)

############################################
#Unseed NTAC example
print("Unseeded NTAC")
#This will ignore the labels, even when provided
nt.solve_unseeded(max_k = 4)
nt.map_partition_to_gt_labels(data.labels) #Use the Hungarian algorithm to map the partition to the ground truth labels
partition = nt.get_partition() #unseeded does not support topk partition
metrics = data.get_metrics(partition, range(data.n), data.labels)
print(f"Accuracy: {metrics['acc']:.3f} ARI: {metrics['ari']:.3f}", f"Weighted F1: {metrics['f1']:.3f}")
```

## Documentation & Examples using the Flywire dataset:

[Link to Docs and further examples](https://benjourdan.github.io/ntac/)