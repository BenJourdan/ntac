#%%
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self, nt, data):
        """
        Parameters:
          nt: an object that provides:
              - nt.get_partition(): returns a 1D array of predicted class labels,
              - nt.embedding: a NumPy array (n_neurons x embedding_dim),
              - nt.reverse_mapping: dictionary mapping bin indices to type names.
          data: an object that must provide data.labels as the ground truth labels.
        """
        self.nt = nt
        self.data = data


    def plot_acc_vs_class_size(self, metrics, bins=None, test_indices=None):
        """
        Plot class accuracy (y) vs. class size (x).

        :param metrics: dict containing "class_acc": {class_label: accuracy}.
        :param bins: if provided, a sequence of bin edges to group class sizes;
                     will plot mean accuracy per bin as bars.
        :param test_indices: optional list/array of node indices.  If given,
                             only classes present in test_indices are shown,
                             and class‐sizes are computed on that subset.
        """
        acc_dict = metrics.get("class_acc", {})
        if not acc_dict:
            print("No class accuracy data available.")
            return

        # 1) choose labels for sizing (test‐only or full)
        if test_indices is not None:
            lbls_for_size = self.data.labels[test_indices]
        else:
            lbls_for_size = self.data.labels

        # 2) get class‐size counts
        unique, counts = np.unique(lbls_for_size, return_counts=True)
        size_dict = dict(zip(unique, counts))

        # 3) collect per‐class (size, acc) for only classes in acc_dict & present in size_dict
        entries = []
        for cls, acc in acc_dict.items():
            if cls in size_dict:
                entries.append((cls, size_dict[cls], acc))

        if bins is None:
            # ——— scatter with aggregation ———
            from collections import defaultdict
            # key = (size, acc), value = sum of test‐counts across all classes with that key
            agg = defaultdict(int)
            for cls, size, acc in entries:
                agg[(size, acc)] += size

            xs, ys, ss = [], [], []
            for (size, acc), total_nodes in agg.items():
                xs.append(size)
                ys.append(acc)
                ss.append(total_nodes)

            # scale marker area so it's visible
            scale = 20
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(xs, ys, s=[s * scale for s in ss],
                       alpha=0.6, edgecolor='black')

            # optional: annotate each point with the total_nodes
            for x, y, total in zip(xs, ys, ss):
                ax.text(x, y, str(total),
                        fontsize=6, ha='center', va='center')

            ax.set_xlabel("Class size (# test nodes)")
            ax.set_ylabel("Accuracy")
            ax.set_title("Per‐Class Accuracy vs. Class Size (aggregated)")

        else:
            # unchanged: bin into bar chart
            sizes = [size for _, size, _ in entries]
            accuracies = [acc for _, _, acc in entries]
            bin_idxs = np.digitize(sizes, bins)
            bin_acc  = []
            bin_cent = []
            for b in range(1, len(bins)):
                idxs = [i for i, bi in enumerate(bin_idxs) if bi == b]
                if not idxs:
                    continue
                bin_acc.append(np.mean([accuracies[i] for i in idxs]))
                bin_cent.append(0.5*(bins[b-1] + bins[b]))

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.bar(bin_cent, bin_acc, width=np.diff(bins), align='center',
                   edgecolor='black')
            ax.set_xlabel("Test‐set class‐size bins")
            ax.set_ylabel("Mean accuracy")
            ax.set_title("Mean Class Accuracy by Class‐Size Bin")

        ax.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.show()



    def plot_class_accuracy(self, metrics):
        """
        Plot a horizontal bar chart of neuron class accuracies.
        :param metrics: Dictionary that contains key "class_acc", a mapping of class -> accuracy.
        """
        acc_dict = metrics.get("class_acc", {})
        if not acc_dict:
            print("No class accuracy data available.")
            return

        # sort classes by descending accuracy
        sorted_acc = sorted(acc_dict.items(), key=lambda x: x[1], reverse=True)
        classes_sorted = [item[0] for item in sorted_acc]
        accuracies_sorted = [item[1] for item in sorted_acc]

        # Dynamically size the figure
        height = max(4, len(classes_sorted) * 0.25)
        fig, ax = plt.subplots(figsize=(10, height))
        y_pos = np.arange(len(classes_sorted))

        ax.barh(y_pos, accuracies_sorted, align='center',
                color='skyblue', edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(classes_sorted, fontsize=8)
        ax.invert_yaxis()  # highest accuracy at top
        ax.set_ylim(-0.5, len(classes_sorted) - 0.5)

        # Add accuracy text labels for each bar
        for i, acc in enumerate(accuracies_sorted):
            ax.text(acc + 0.01, y_pos[i], f"{acc:.2f}", va="center", fontsize=8)

        ax.set_xlabel("Accuracy", fontsize=12)
        ax.set_title("Neuron Class Accuracies (Sorted)", fontsize=14)

        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, labels=None, normalize=False, cmap='Blues', fscore_threshold=1.0, include_labels=None):
        """
        Plot a confusion matrix comparing true and predicted class labels.

        :param labels: list of all possible classes (for ordering). If None, derived from union of true and predicted.
        :param normalize: if True, normalize each row to sum to 1.
        :param cmap: colormap for the matrix.
        :param fscore_threshold: show only labels whose F1 score is below this threshold.
        :param include_labels: optional list or set of labels to include regardless of F1; used to restrict final plotted labels.
        """
        # Retrieve true and predicted labels
        y_true = self.data.labels
        y_pred = self.nt.get_partition()

        # Determine full label list
        if labels is None:
            labels = sorted(set(y_true) | set(y_pred))

        # Compute F1 scores per class
        f1_scores = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

        # Apply F1 threshold filter
        labels_filtered = [label for label, f1 in zip(labels, f1_scores) if f1 < fscore_threshold]

        # Apply include_labels restriction if provided
        if include_labels is not None:
            include_labels = set(include_labels)
            labels_filtered = [label for label in labels_filtered if label in include_labels]

        if not labels_filtered:
            print("No labels to plot after applying filters.")
            return

        # Compute confusion matrix with filtered labels
        cm = confusion_matrix(y_true, y_pred, labels=labels_filtered)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

        # Plot the matrix
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)

        tick_marks = np.arange(len(labels_filtered))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(labels_filtered, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(labels_filtered, fontsize=8)

        ax.set_ylabel('True label', fontsize=12)
        ax.set_xlabel('Predicted label', fontsize=12)
        title = 'Confusion Matrix'
        if normalize:
            title += ' (Normalized)'
        title += f' (F1 < {fscore_threshold})'
        if include_labels is not None:
            title += f' ∩ include_labels'
        ax.set_title(title, fontsize=14)

        plt.tight_layout()
        plt.show()


    def plot_embedding_comparison(self, class_name1, class_name2, show_error=False, use_gt_labels=False, min_threshold=0.0):
        """
        Plot a back-to-back horizontal bar chart comparing in- and out-degree embeddings
        for two classes. If the two class names are the same, the bars will align exactly.
        
        Assumes:
          - self.nt.get_partition() returns predicted labels.
          - self.nt.embedding is (n_neurons, embedding_dim) with the first half corresponding
            to in-degree and the second half to out-degree.
          - self.nt.reverse_mapping maps bin indices (0 to half_dim-1) to type names.
        """
        if use_gt_labels:
            # Use ground truth labels if specified.
            partition = self.data.labels
        else:
            partition = self.nt.get_partition()
        indices1 = np.where(partition == class_name1)[0] if class_name1 else np.array([])
        indices2 = np.where(partition == class_name2)[0] if class_name2 else np.array([])


        unique_classes = set(partition)
        print(unique_classes)

        if len(indices1) == 0 and len(indices2) == 0:
            raise ValueError("No neurons found for the given class names.")

        embedding_dim = self.nt.embedding.shape[1]
        half_dim = embedding_dim // 2

        # Helper function to compute in/out means and stds.
        def compute_stats(indices):
            if len(indices) == 0:
                return None, None, None, None
            embs = self.nt.embedding[indices]  # shape: (N, embedding_dim)
            in_deg = embs[:, :half_dim]
            out_deg = embs[:, half_dim:]
            return (np.mean(in_deg, axis=0), np.std(in_deg, axis=0),
                    np.mean(out_deg, axis=0), np.std(out_deg, axis=0))

        in1_avg, in1_std, out1_avg, out1_std = compute_stats(indices1)
        in2_avg, in2_std, out2_avg, out2_std = compute_stats(indices2)

        labels = [self.nt.reverse_mapping[i] for i in range(half_dim)]
        y_positions = np.arange(half_dim)


        # Apply threshold filtering
        if min_threshold > 0:
            mask = np.ones(half_dim, dtype=bool)
            for i in range(half_dim):
                v1 = in1_avg[i] if in1_avg is not None else 0
                v2 = in2_avg[i] if in2_avg is not None else 0
                if abs(v1) < min_threshold and abs(v2) < min_threshold:
                    mask[i] = False

            # Filter all relevant arrays and labels
            labels = [label for i, label in enumerate(labels) if mask[i]]
            y_positions = np.arange(len(labels))  # reindex compactly


            if in1_avg is not None:
                in1_avg = in1_avg[mask]
                in1_std = in1_std[mask]
                out1_avg = out1_avg[mask]
                out1_std = out1_std[mask]
            if in2_avg is not None:
                in2_avg = in2_avg[mask]
                in2_std = in2_std[mask]
                out2_avg = out2_avg[mask]
                out2_std = out2_std[mask]
            half_dim = len(y_positions)
        # Create the plot
        fig, ax = plt.subplots(figsize=(8, 8))

        # Set bar height and offsets. If both classes are the same use zero offsets.
        if class_name1 == class_name2:
            bar_height = 0.3
            offsets = {'c1_in': 0, 'c1_out': 0, 'c2_in': 0, 'c2_out': 0}
        else:
            bar_height = 0.4
            offsets = {
                'c1_in':  0.5 * bar_height,
                'c1_out': -0.5 * bar_height,
                'c2_in':  0.5 * bar_height,
                'c2_out': -0.5 * bar_height
            }

        def barh_with_optional_error(ax, y, width, error, color, label=None):
            if show_error and error is not None:
                ax.barh(y, width, xerr=error, height=bar_height,
                        color=color, edgecolor='black', capsize=4, label=label)
            else:
                ax.barh(y, width, height=bar_height,
                        color=color, edgecolor='black', label=label)

        # Plot Class 1 in-degree (positive, to the right)
        if in1_avg is not None:
            for i in range(half_dim):
                lbl = f"{class_name1} (In)" if i == 0 else None
                barh_with_optional_error(ax, y_positions[i] + offsets['c1_in'], in1_avg[i],
                                         in1_std[i], color='steelblue', label=lbl)

        # Plot Class 1 out-degree (positive, to the right)
        if out1_avg is not None:
            for i in range(half_dim):
                lbl = f"{class_name1} (Out)" if i == 0 else None
                barh_with_optional_error(ax, y_positions[i] + offsets['c1_out'], out1_avg[i],
                                         out1_std[i], color='lightskyblue', label=lbl)

        # Plot Class 2 in-degree (mirrored to the left)
        if in2_avg is not None:
            for i in range(half_dim):
                lbl = f"{class_name2} (In)" if i == 0 else None
                barh_with_optional_error(ax, y_positions[i] + offsets['c2_in'], -in2_avg[i],
                                         in2_std[i], color='salmon', label=lbl)

        # Plot Class 2 out-degree (mirrored to the left)
        if out2_avg is not None:
            for i in range(half_dim):
                lbl = f"{class_name2} (Out)" if i == 0 else None
                barh_with_optional_error(ax, y_positions[i] + offsets['c2_out'], -out2_avg[i],
                                         out2_std[i], color='lightcoral', label=lbl)

        ax.axvline(0, color='black', linewidth=1)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels)
        ax.set_ylim(-1, half_dim)
        ax.set_xlabel("Average Embedding Value")
        ax.set_ylabel("Type")
        title = "Back-to-Back In/Out Degree Comparison"
        if class_name1:
            title += f" | Class1: {class_name1}"
        if class_name2:
            title += f" vs Class2: {class_name2}"
        ax.set_title(title)

        ax.grid(axis='x', linestyle='--', alpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Deduplicate legend entries
        handles, leg_labels = ax.get_legend_handles_labels()
        by_label = dict(zip(leg_labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='best')

        # Mirror the x-axis tick labels to show only positive numbers.
        xticks = ax.get_xticks()
        ax.set_xticklabels([f"{abs(x):.1f}" for x in xticks])

        plt.tight_layout()
        plt.show()

    def plot_true_label_histogram(self, alg_class, top_k=None):
        """
        Plot a histogram of the ground truth labels among neurons that
        are assigned to a given predicted class (alg_class). If top_k is
        provided, only the top k most common labels will be shown.
        
        Assumes:
          - self.nt.get_partition() returns predicted class labels.
          - self.data.labels contains the ground truth labels.
        """
        partition = self.nt.get_partition()    # predicted labels
        true_labels = self.data.labels           # ground truth labels
        
        # Indices corresponding to the predicted class.
        indices = np.where(partition == alg_class)[0]
        if len(indices) == 0:
            raise ValueError(f"No neurons found for predicted class '{alg_class}'.")
        
        selected_true_labels = true_labels[indices]
        unique_labels, counts = np.unique(selected_true_labels, return_counts=True)
        
        # Sort in descending order (largest counts first).
        order = np.argsort(-counts)
        unique_labels = unique_labels[order]
        counts = counts[order]
        
        # Apply top_k filtering if provided.
        if top_k is not None:
            unique_labels = unique_labels[:top_k]
            counts = counts[:top_k]
        
        # Create the histogram.
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(unique_labels, counts, color='skyblue', edgecolor='black')
        ax.set_xlabel("True (Real) Classes")
        ax.set_ylabel("Count")
        ax.set_title(f"Histogram of True Labels within Predicted Class '{alg_class}'")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


# Example usage:
# Assuming nt and data objects are defined in your notebook:
# vis = Visualizer(nt, data)
# vis.plot_class_accuracy(metrics)
# vis.plot_embedding_comparison("R7", "R8", show_error=False)
# vis.plot_true_label_histogram("T4a", top_k=5)
