import random

import numpy as np
from matplotlib import pyplot as plt


class Plotter:
    def __init__(self, classes):
        self.colors = self.get_colors(classes)
        self.markers = self.get_markers(classes)

    def get_colors(self, classes):
        colors = list()
        for clazz in range(classes):
            color = "#%06x" % random.randint(0, 0xFFFFFF)
            colors.append(color)
        return colors

    def get_markers(self, classes):
        markers = list()
        for clazz in range(classes):
            marker = random.choice(['o', 's', 'v', '^', '<', '>', '8', 'P', 'p', 'h', '+', 'x'])
            markers.append(marker)
        return markers

    def plot_2d_space(self, X, y, title='Classes'):
        for index, label in enumerate(np.unique(y)):
            plt.scatter(
                X[y == label, 0],
                X[y == label, 1],
                c=self.colors[index], label=label, marker=self.markers[index]
            )
        plt.title(title)
        plt.legend(loc='upper right')
        plt.show()
