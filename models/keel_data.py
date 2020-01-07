from sklearn.model_selection import train_test_split

import src.plot as plot
from models.testable_keel_data import TestableKeelData


class KeelData:
    def __init__(self, file_name, size, features, classes, x, y):
        self.file_name = file_name
        self.size = size
        self.features = features
        self.classes = classes
        self.x = x
        self.y = y

    def print_info(self):
        print('Keel Dataset Info:')
        print('==> File Name: %s' % self.file_name)
        print('==> Size: %s' % self.size)
        print('==> Features: %s' % self.features)
        print('==> Classes: %s' % self.classes)

    def plot_class_distribution(self):
        plot.plot_2d_space(self.x, self.y, title="Dataset class distribution")

    def as_testable(self, test_size=0.25, random_state=0):
        x_train, x_test, y_train, y_test = train_test_split(
            self.x,
            self.y,
            test_size=test_size,
            random_state=random_state
        )

        return TestableKeelData(self.file_name, self.size, self.features, self.classes, test_size, x_train, x_test,
                                y_train, y_test)
