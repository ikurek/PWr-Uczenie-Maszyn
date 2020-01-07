import random

import numpy as np
from matplotlib import pyplot as plt


def plot_2d_space(X, y, title='Classes'):
    markers = ['o', 's', 'v', '^', '<', '>', '8', 'P', 'p', 'h', '+', 'x']
    for label in np.unique(y):
        color = "#%06x" % random.randint(0, 0xFFFFFF)
        marker = random.choice(markers)
        plt.scatter(
            X[y == label, 0],
            X[y == label, 1],
            c=color, label=label, marker=marker
        )
    plt.title(title)
    plt.legend(loc='upper right')
    plt.show()
