import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from models.classification_statistics import ClassificationStatistics
from models.testable_keel_data import TestableKeelData


# FIXME: Returns bad metrics for multiclass problems
class CommonClassification:
    def __base_classification(self, testable_keel_dataset, classifier):
        classifier.fit(testable_keel_dataset.x_train, testable_keel_dataset.y_train)
        y_predicted = classifier.predict(testable_keel_dataset.x_test)
        return self.__get_metrics(
            testable_keel_dataset.file_name,
            testable_keel_dataset.size,
            testable_keel_dataset.classes,
            testable_keel_dataset.y_test,
            y_predicted
        )

    def __get_metrics(self, file_name, size, classes, y_true, y_predicted):
        accuracy = metrics.accuracy_score(y_true, y_predicted)
        confusion_matrix = metrics.confusion_matrix(y_true, y_predicted)
        if classes > 2:
            labels = np.unique(y_true)
            average = 'micro'
            f1 = metrics.f1_score(y_true, y_predicted, labels=labels, average=average)
            recall = metrics.recall_score(y_true, y_predicted, labels=labels, average=average)
            precision = metrics.precision_score(y_true, y_predicted, labels=labels, average=average)
        else:
            positive_label = 'positive'
            f1 = metrics.f1_score(y_true, y_predicted, pos_label=positive_label)
            recall = metrics.recall_score(y_true, y_predicted, pos_label=positive_label)
            precision = metrics.precision_score(y_true, y_predicted, pos_label=positive_label)
        return ClassificationStatistics(file_name, size, accuracy, precision, confusion_matrix, f1, recall)

    def naive_bayes_gaussian_classification(self, testable_keel_dataset: TestableKeelData):
        naive_bayes_gaussian_classifier = GaussianNB()
        result = self.__base_classification(testable_keel_dataset, naive_bayes_gaussian_classifier)
        return result

    def decision_tree_classification(self, testable_keel_dataset: TestableKeelData):
        decision_tree_classifier = DecisionTreeClassifier()
        result = self.__base_classification(testable_keel_dataset, decision_tree_classifier)
        return result

    def random_forest_classification(self, testable_keel_dataset: TestableKeelData):
        random_forest_classifier = RandomForestClassifier()
        result = self.__base_classification(testable_keel_dataset, random_forest_classifier)
        return result
