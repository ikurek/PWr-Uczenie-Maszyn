from src.classification import CommonClassification
from src.preprocessing import CommonPreprocessing
from src.read_data import DataReader

SMOTE_K_NEIGHBOURS = 3
DATA_DIR = "../data"

data_reader = DataReader(DATA_DIR)
common_classifier = CommonClassification()
common_preprocessing = CommonPreprocessing()

keel_data = data_reader.read_keel_dat_directory()

for data in keel_data:
    # initialize file names and params
    print("Current dataset: " + str(data.file_name))
    csv_export_file_name = "../results/smote_oversampling_k3/data-naive_bayes.csv"
    output_csv = open(csv_export_file_name, "a")
    confusion_matrix_plot_file_name = "../results/smote_oversampling_k3/confusion_matrix/%s-%s.%s" % (
        data.file_name, "naive_bayes", "png")
    class_distribution_plot_file_name = "../results/smote_oversampling_k3/class_distribution/%s-%s.%s" % (
        data.file_name, "naive_bayes", "png")

    # Split dataset and perform classification
    testable = data.as_testable()
    sampled_testable = common_preprocessing.smote_over_sampling(testable, SMOTE_K_NEIGHBOURS)
    bayes_result = common_classifier.naive_bayes_gaussian_classification(sampled_testable)

    # Print plots and data to files
    print("%s;%s;%s;%s;%s" % (
        bayes_result.file_name, bayes_result.accuracy, bayes_result.precision, bayes_result.recall, bayes_result.f1),
          file=output_csv)
    bayes_result.plot_confusion_matrix(filename=confusion_matrix_plot_file_name)
    sampled_testable.plot_train_class_distribution(filename=class_distribution_plot_file_name)
    output_csv.close()

for data in keel_data:
    # initialize file names and params
    csv_export_file_name = "../results/smote_oversampling_k3/data-random_forest.csv"
    output_csv = open(csv_export_file_name, "a")
    confusion_matrix_plot_file_name = "../results/smote_oversampling_k3/confusion_matrix/%s-%s.%s" % (
        data.file_name, "random_forest", "png")
    class_distribution_plot_file_name = "../results/smote_oversampling_k3/class_distribution/%s-%s.%s" % (
        data.file_name, "random_forest", "png")

    # Split dataset and perform classification
    testable = data.as_testable()
    sampled_testable = common_preprocessing.smote_over_sampling(testable, SMOTE_K_NEIGHBOURS)
    random_forest_result = common_classifier.random_forest_classification(sampled_testable)

    # Print plots and data to files
    print("%s;%s;%s;%s;%s" % (
        random_forest_result.file_name, random_forest_result.accuracy, random_forest_result.precision,
        random_forest_result.recall, random_forest_result.f1),
          file=output_csv)
    random_forest_result.plot_confusion_matrix(filename=confusion_matrix_plot_file_name)
    sampled_testable.plot_train_class_distribution(filename=class_distribution_plot_file_name)
    output_csv.close()

for data in keel_data:
    # initialize file names and params
    csv_export_file_name = "../results/smote_oversampling_k3/data-decision_tree.csv"
    output_csv = open(csv_export_file_name, "a")
    confusion_matrix_plot_file_name = "../results/smote_oversampling_k3/confusion_matrix/%s-%s.%s" % (
        data.file_name, "decision_tree", "png")
    class_distribution_plot_file_name = "../results/smote_oversampling_k3/class_distribution/%s-%s.%s" % (
        data.file_name, "decision_tree", "png")

    # Split dataset and perform classification
    testable = data.as_testable()
    sampled_testable = common_preprocessing.smote_over_sampling(testable, SMOTE_K_NEIGHBOURS)
    decision_tree_result = common_classifier.decision_tree_classification(sampled_testable)

    # Print plots and data to files
    print("%s;%s;%s;%s;%s" % (
        decision_tree_result.file_name, decision_tree_result.accuracy, decision_tree_result.precision,
        decision_tree_result.recall, decision_tree_result.f1),
          file=output_csv)
    decision_tree_result.plot_confusion_matrix(filename=confusion_matrix_plot_file_name)
    sampled_testable.plot_train_class_distribution(filename=class_distribution_plot_file_name)
    output_csv.close()
