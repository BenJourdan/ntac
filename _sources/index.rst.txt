.. ntac documentation master file, created by
   sphinx-quickstart on Wed Apr  2 11:41:50 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

NTAC documentation
==================


This is a Python implementation of the Neuronal typing algorithm described in

`Gregory Schwartzman, Ben Jourdan, David García-Soriano, Arie Matsliah. Connectivity Is All You Need: Inferring Neuronal Types with NTAC. bioRxiv. 2025 <https://www.biorxiv.org/content/10.1101/2025.06.11.659184v1>`_

NTAC (Neuronal Type Assignment from Connectivity) groups neurons into cell types based solely on synaptic connectivity. It comes in two variants:

- **Seeded (semi-supervised):** Requires a small fraction of neurons with known labels.  
- **Unseeded (unsupervised):** Requires no labels.


Installation:
============================

To install the `ntac` package, you can use pip:

.. code-block:: bash

   pip install ntac

Optionally, install cudatoolkit with conda to speed up unseeded NTAC:

.. code-block:: bash

   conda install -c conda-forge cudatoolkit


Quickstart with NTAC
============================

.. code-block:: python

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


Flywire Examples:
============================
See the examples for using NTAC in seeded and unseeded modes with the Flywire dataset:

- `Seeded NTAC <seeded_example.html>`_
- `Unseeded NTAC <unseeded_example.html>`_

.. toctree::
   :maxdepth: 2
   :hidden:

   seeded_example
   unseeded_example
   api