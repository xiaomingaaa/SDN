import numpy as np

def calculate_auc_roc(labels, predictions):
    sorted_indices = np.argsort(predictions)[::-1]
    sorted_predictions = predictions[sorted_indices]
    sorted_labels = labels[sorted_indices]

    
    num_positive = np.sum(sorted_labels == 1)
    num_negative = len(sorted_labels) - num_positive

    tpr = np.cumsum(sorted_labels) / num_positive
    fpr = np.cumsum(1 - sorted_labels) / num_negative

    auc_roc = np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1])) / 2

    return auc_roc


def calculate_aupr(labels, predictions):
    sorted_indices = np.argsort(predictions)[::-1]
    sorted_predictions = predictions[sorted_indices]
    sorted_labels = labels[sorted_indices]

    precision = np.cumsum(sorted_labels) / np.arange(1, len(sorted_labels) + 1)
    recall = np.cumsum(sorted_labels) / np.sum(sorted_labels)

    aupr = np.sum((recall[1:] - recall[:-1]) * precision[1:])

    return aupr
