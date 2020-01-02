import random

from sklearn.model_selection import train_test_split

from models.testable_keel_data import TestableKeelData


class KeelData:
    def __init__(self, file_name, size, features, imbalance_ratio, x, y, xy):
        self.file_name = file_name
        self.size = size
        self.features = features
        self.imbalance_ratio = imbalance_ratio
        self.x = x
        self.y = y
        self.xy = xy

    def print_info(self):
        print('Keel Dataset Info:')
        print('==> File Name: %s' % self.file_name)
        print('==> Size: %s' % self.size)
        print('==> Features: %s' % self.features)
        print('==> Imbalance Ratio: %s' % self.imbalance_ratio)
        print('==> Sample Entry: %s' % random.choice(self.xy))

    def as_testable(self, test_size=0.25, random_state=0):
        x_train, x_test, y_train, y_test = train_test_split(
            self.x,
            self.y,
            test_size=test_size,
            random_state=random_state
        )

        return TestableKeelData(self.file_name, self.size, self.features, test_size, x_train, x_test, y_train, y_test)
