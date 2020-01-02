import random

from src.classification import CommonClassification
from src.read_data import DataReader

DATA_DIR = "../data"
data_reader = DataReader(DATA_DIR)
common_classifier = CommonClassification()

keel_data = data_reader.read_keel_dat_directory()

random_keel_data = random.choice(keel_data)
random_keel_data.print_info()

random_testable_keel_data = random_keel_data.as_testable()
random_testable_keel_data.print_info()

classification_result = common_classifier.naive_bayes_gaussian_classification(random_testable_keel_data)
print(classification_result.accuracy)

classification_result = common_classifier.decision_tree_classification(random_testable_keel_data)
print(classification_result.accuracy)

classification_result = common_classifier.random_forest_classification(random_testable_keel_data)
print(classification_result.accuracy)
