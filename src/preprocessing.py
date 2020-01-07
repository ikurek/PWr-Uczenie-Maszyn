import copy

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids

from models.testable_keel_data import TestableKeelData


# FIXME: SMOTE/ADASYN Fail sometimes, bad KNN parameters
class CommonPreprocessing:
    def __base_preprocessing(self, testable_keel_dataset: TestableKeelData, sampler):
        dataset = copy.deepcopy(testable_keel_dataset)
        sampled_test_x, sampled_test_y = sampler.fit_resample(dataset.x_test, dataset.y_test)
        sampled_train_x, sampled_train_y = sampler.fit_resample(dataset.x_train, dataset.y_train)
        dataset.update_with_datasets(sampled_train_x, sampled_test_x, sampled_train_y, sampled_test_y)
        return dataset

    def random_under_sampling(self, keel_dataset):
        sampler = RandomUnderSampler()
        result = self.__base_preprocessing(keel_dataset, sampler)
        return result

    def cluster_centroids_under_sampling(self, keel_dataset):
        sampler = ClusterCentroids()
        result = self.__base_preprocessing(keel_dataset, sampler)
        return result

    def random_over_sampling(self, keel_dataset):
        sampler = RandomOverSampler()
        result = self.__base_preprocessing(keel_dataset, sampler)
        return result

    def smote_over_sampling(self, keel_dataset):
        sampler = SMOTE()
        result = self.__base_preprocessing(keel_dataset, sampler)
        return result

    def adasyn_over_sampling(self, keel_dataset):
        sampler = ADASYN()
        result = self.__base_preprocessing(keel_dataset, sampler)
        return result
