from src.classification import CommonClassification
from src.preprocessing import CommonPreprocessing
from src.read_data import DataReader

DATA_DIR = "../data"
ATTEMPTS = 10
TRAIN_TEST_SPLIT_RATIO = 0.25

data_reader = DataReader(DATA_DIR)
common_classifier = CommonClassification()
common_preprocessing = CommonPreprocessing()

keel_data = data_reader.read_keel_dat_file("glass.dat")

random_forest_results = list()
decision_tree_results = list()
naive_bayes_results = list()

for attempt in range(ATTEMPTS):
    testable_data = keel_data.as_testable(test_size=TRAIN_TEST_SPLIT_RATIO)
    preprocessed_data = common_preprocessing.random_over_sampling(testable_data)

    random_forest_result = common_classifier.random_forest_classification(preprocessed_data)
    decision_tree_result = common_classifier.decision_tree_classification(preprocessed_data)
    naive_bayes_result = common_classifier.naive_bayes_gaussian_classification(preprocessed_data)

    random_forest_results.append(random_forest_result)
    decision_tree_results.append(decision_tree_result)
    naive_bayes_results.append(naive_bayes_result)

print("end")
