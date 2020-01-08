import random

from src.classification import CommonClassification
from src.preprocessing import CommonPreprocessing
from src.read_data import DataReader

DATA_DIR = "../data"
data_reader = DataReader(DATA_DIR)
common_classifier = CommonClassification()
common_preprocessing = CommonPreprocessing()

all_keel_data = data_reader.read_keel_dat_directory()
keel_data = random.choice(all_keel_data)

random_testable_keel_data = keel_data.as_testable()
print("Testable:")
random_testable_keel_data.print_info()
random_testable_keel_data.plot_train_class_distribution()

classification_result = common_classifier.naive_bayes_gaussian_classification(random_testable_keel_data)
classification_result.print_info()
classification_result.plot_confusion_matrix()
