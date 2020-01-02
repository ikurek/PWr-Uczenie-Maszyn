from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from models.classification_statistics import ClassificationStatistics
from models.testable_keel_data import TestableKeelData


class CommonClassification:
    def __base_classification(self, testable_keel_dataset, classifier):
        classifier.fit(testable_keel_dataset.x_train, testable_keel_dataset.y_train)
        y_prediction = classifier.predict(testable_keel_dataset.x_test)
        accuracy = metrics.accuracy_score(testable_keel_dataset.y_test, y_prediction)
        return ClassificationStatistics(accuracy)

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
