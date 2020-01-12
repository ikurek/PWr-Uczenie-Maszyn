import numpy as np


def get_imbalance_ratio(dataset_y):
    (values, counts) = np.unique(dataset_y, return_counts=True)
    most_common_index = np.argmax(counts)
    least_common_index = np.argmin(counts)
    return counts[most_common_index] / counts[least_common_index]
