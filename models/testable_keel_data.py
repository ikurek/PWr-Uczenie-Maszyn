from src.plot import Plotter


class TestableKeelData:
    def __init__(self, file_name, size, features, classes, test_train_split_ratio, x_train, x_test, y_train, y_test):
        self.file_name = file_name
        self.size = size
        self.features = features
        self.classes = classes
        self.test_train_split_ratio = test_train_split_ratio
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.plotter = Plotter(classes)

    def update_with_datasets(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.size = len(x_train) + len(x_test)
        self.test_train_split_ratio = len(x_test) / len(x_train)

    def print_info(self):
        print('Keel Testable Dataset Info:')
        print('==> File Name: %s' % self.file_name)
        print('==> Size: %s' % self.size)
        print('==> Features: %s' % self.features)
        print('==> Classes: %s' % self.classes)
        print('==> Test-Train Split Ratio: %s' % self.test_train_split_ratio)
        print('==> Train Size: %s' % len(self.x_train))
        print('==> Test Size: %s' % len(self.x_test))

    def plot_train_class_distribution(self, filename=''):
        self.plotter.plot_2d_space(self.x_train, self.y_train, title="Train set class distribution", filename=filename)
