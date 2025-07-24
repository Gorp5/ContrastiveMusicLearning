from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score
import numpy as np

def visualize_ROC_PR_AUC(probs, labels):
    # Store per-threshold values
    thresholds = np.linspace(0, 1, 200)
    precision_all = []
    recall_all = []
    roc_aucs = []
    pr_aucs = []

    for class_idx in range(probs.shape[1]):
        y_true = labels[:, class_idx]
        y_score = probs[:, class_idx]

        precision, recall, thresh = precision_recall_curve(y_true, y_score)

        # Interpolate to get precision/recall at uniform thresholds
        interp_precision = np.interp(thresholds, thresh, precision[:-1])  # precision[:-1] because it's len(thresh)+1
        interp_recall = np.interp(thresholds, thresh, recall[:-1])

        precision_all.append(interp_precision)
        recall_all.append(interp_recall)

        if len(np.unique(y_true)) > 1:
            roc_auc = roc_auc_score(y_true, y_score)
            pr_auc = average_precision_score(y_true, y_score)
            roc_aucs.append(roc_auc)
            pr_aucs.append(pr_auc)

    # Average across all classes
    precision_mean = np.mean(precision_all, axis=0)
    recall_mean = np.mean(recall_all, axis=0)
    roc_auc_macro = np.mean(roc_aucs) if roc_aucs else float('nan')
    pr_auc_macro = np.mean(pr_aucs) if pr_aucs else float('nan')

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, precision_mean, label='Precision')
    plt.plot(thresholds, recall_mean, label='Recall')
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Precision and Recall vs Threshold (Macro-Averaged over Genres)")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"ROC-AUC: {roc_auc_macro:.4f}\tPR-AUC: {pr_auc_macro:.4f}")