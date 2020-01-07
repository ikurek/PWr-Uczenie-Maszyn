class ClassificationStatistics:
    def __init__(self, file_name, size, accuracy, precision, confusion_matrix, f1, recall):
        self.file_name = file_name
        self.size = size
        self.accuracy = accuracy
        self.precision = precision
        self.confusion_matrix = confusion_matrix
        self.f1 = f1
        self.recall = recall

    def print_info(self):
        print('Clasification Statistics:')
        print('==> File Name: %s' % self.file_name)
        print('==> Size: %s' % self.size)
        print('==> Accuracy: %s' % self.accuracy)
        print('==> Precision: %s' % self.precision)
        print('==> Confusion Matrix: %s' % self.confusion_matrix)
        print('==> F1: %s' % self.f1)
        print('==> Recall: %s' % self.recall)
