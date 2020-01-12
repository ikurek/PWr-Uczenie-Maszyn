from sklearn.model_selection import train_test_split

import src.imbalance_ratio as ir
from models.testable_keel_data import TestableKeelData
from src.plot import Plotter


class KeelData:
    def __init__(self, file_name, size, features, classes, x, y):
        self.file_name = file_name
        self.size = size
        self.features = features
        self.classes = classes
        self.x = x
        self.y = y
        self.imbalance_ratio = ir.get_imbalance_ratio(y)
        self.plotter = Plotter(classes)

    def print_info(self):
        print('Keel Dataset Info:')
        print('==> File Name: %s' % self.file_name)
        print('==> Size: %s' % self.size)
        print('==> Features: %s' % self.features)
        print('==> Classes: %s' % self.classes)
        print('==> Imbalance Ratio: %s' % self.imbalance_ratio)

    def plot_class_distribution(self, filename=''):
        self.plotter.plot_2d_space(self.x, self.y, title="Dataset class distribution", filename=filename)

    def as_testable(self, test_size=0.25):
        x_train, x_test, y_train, y_test = train_test_split(
            self.x,
            self.y,
            stratify=self.y,
            shuffle=True
        )

        return TestableKeelData(self.file_name, self.size, self.features, self.classes, test_size, x_train, x_test,
                                y_train, y_test)
