import numpy as np
from sklearn.metrics import roc_curve, auc, average_precision_score
def calculate_ood_metrics(scores, labels,total_count):
    """
    Calculate OOD detection metrics: AUROC, AUPR, and FPR@TPR95.
    Args:
        scores (np.ndarray): Anomaly scores for all points (both in-distribution and OOD).
        labels (np.ndarray): Binary labels (0 for in-distribution, 1 for OOD).
    Returns:
        auroc (float): Area Under the ROC Curve.
        aupr (float): Area Under the Precision-Recall Curve.
        fpr95 (float): False Positive Rate at 95% True Positive Rate.
    """
    # Ensure inputs are numpy arrays
    scores = np.array(scores)
    labels = np.array(labels)
    fix_count = int(total_count)

    # Compute AUPR
    aupr = compute_aupr(labels, scores, fix_count)

    return aupr




def compute_aupr(labels, scores, fix_count):
    """
    Compute the Area Under the Precision-Recall Curve (AUPR) manually.
    """
    # Sort scores in descending order
    sorted_indices = np.argsort(scores)[::-1]
    sorted_scores = scores[sorted_indices]
    sorted_labels = labels[sorted_indices]

    # Count total positives and negatives
    num_positives = np.sum(labels == 1) - fix_count 
    num_negatives = np.sum(labels == 0) + fix_count

    # Initialize variables
    precision = []
    recall = []
    tp = 0  # True Positives
    fp = 0  # False Positives

    for i in range(len(sorted_scores)):
        if sorted_labels[i] == 1:
            tp += 1
        else:
            fp += 1
        precision.append(tp / (tp + fp))  
        recall.append(tp / num_positives)
    aupr_area = auc(recall, precision)
    return aupr_area



