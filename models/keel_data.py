class KeelData:
    def __init__(self, file_name, rows, features, imbalance_ratio, x, y, xy):
        self.file_name = file_name
        self.rows = rows
        self.features = features
        self.imbalance_ratio = imbalance_ratio
        self.x = x
        self.y = y
        self.xy = xy

    def print_info(self):
        print('Keel %s Dataset Info:' % self.file_name)
        print('==> Rows: %s' % self.rows)
        print('==> Features: %s' % self.features)
        print('==> Imbalance ratio: %s' % self.imbalance_ratio)
        print('==> Sample entry: %s' % self.xy[0])

