class TestableKeelData:
    def __init__(self, file_name, size, features, test_train_split_ratio, x_train, x_test, y_train, y_test):
        self.file_name = file_name
        self.size = size
        self.features = features
        self.classes = 2
        self.test_train_split_ratio = test_train_split_ratio
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def print_info(self):
        print('Keel Testable Dataset Info:')
        print('==> File Name: %s' % self.file_name)
        print('==> Size: %s' % self.size)
        print('==> Features: %s' % self.features)
        print('==> Classes: %s' % self.classes)
        print('==> Test-Train Split Ratio: %s' % self.test_train_split_ratio)
        print('==> Train Size: %s' % len(self.x_train))
        print('==> Test Size: %s' % len(self.x_test))
