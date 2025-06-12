Seeded NTAC example
==========================

This example shows how to use NTAC in seeded mode, with the Flywire dataset.

.. code-block:: python

    import ntac
    import os
    from os.path import join

    from ntac import Ntac, FAFBData

    # first download the flywire data we'll need for this example. This will be cached in ~/.ntac/flywire_data
    ntac.download_flywire_data(verbose=True)


    # SeededNtac is happy to accept an aribtrary CSR adjacency matrix and a list of labels
    # where ? corresponds to unlabeled nodes. In this example we will use 
    # the flywire dataset, for which we have a custom data loader.

    # This is where the data lives:
    base_path = join(os.path.expanduser("~"), ".ntac")


    # now we decide which part of the flywire dataset we want to use:
    side = "left"
    gender = "female"

    area = "ol_columnar" #other options: ol_intrinsic, entire_visual_system, central_brain, entire_brain


    ignore_side = False
    if area == "central_brain" or area == "entire_brain":
        side = "left_and_right"
    if area == "entire_brain_and_nerve_cord":
        side = "left_and_right"
        ignore_side = True


    data_path = join(base_path,f"flywire_data/dynamic_data/{gender}_brain/{area}")
    edges_file = f"{data_path}/{side}_edges.csv"
    types_file = f"{data_path}/{side}_clusters.csv"

    print("Loading flywire data")



    data = FAFBData(edges_file=edges_file, types_file=types_file, ignore_side=ignore_side)


    random_seed = 5

    train_indices, test_indices = data.test_train_split(train_size=0.05, sampling_type="at_least_one_per_class", random_seed=random_seed)

    #Artifically set test labels to "?"
    labels = data.labels.copy()
    labels[test_indices] = data.unlabeled_symbol




    nt = Ntac(data = data, labels = labels, lr=1, topk=1, verbose=False)
    num_iters = 15
    for i in range(num_iters):
        print(f"Step {i}")
        nt.step()
        final_partition = nt.get_partition()
        metrics = data.get_metrics(final_partition, test_indices, data.labels, compute_class_acc= (i == num_iters - 1))
        print(f"Accuracy: {metrics['acc']:.3f} ARI: {metrics['ari']:.3f}", f"Weighted F1: {metrics['f1']:.3f}")




