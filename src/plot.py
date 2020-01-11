import random

import numpy as np
import pandas as pd
import seaborn as sn
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

    def plot_2d_space(self, X, y, title='Classes', filename=""):
        for index, label in enumerate(np.unique(y)):
            plt.scatter(
                X[y == label, 0],
                X[y == label, 1],
                c=self.colors[index], label=label, marker=self.markers[index]
            )
        plt.title(title)
        plt.legend(loc='upper right')
        if len(filename) > 0:
            plt.savefig(filename)
        plt.show()

    def plot_confusion_matrix(self, confusion_matrix, classes, filename=""):
        df_cm = pd.DataFrame(confusion_matrix, classes, classes)
        sn.set(font_scale=1.4)  # for label size
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 12})  # font size
        if len(filename) > 0:
            plt.savefig(filename)
        plt.show()
