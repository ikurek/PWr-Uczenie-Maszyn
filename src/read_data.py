import numpy as np
import pandas as pd
import glob
from os import listdir, path
from models.keel_data import KeelData


class DataReader:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def read_keel_dat_file(self, file_name: str, skip=13):
        file_path = self.data_dir + "/" + file_name
        data_file = pd.read_csv(file_path, skiprows=skip, header=None)
        data_file = data_file.rename(columns={data_file.columns[len(list(data_file)) - 1]: 'Class'})
        mapping = {'positive': 1, 'negative': 0}
        data_file = data_file.replace({'Class': mapping}, regex=True)
        data_file_positive = data_file[data_file['Class'] == 0]
        data_file_negative = data_file[data_file['Class'] == 1]
        xy = data_file.values.astype(np.float32)
        y = xy[:, xy.shape[1] - 1]
        x = np.delete(xy, xy.shape[1] - 1, axis=1)
        rows = len(data_file)
        features = (len(list(data_file)) - 1)
        imbalance_ratio = float(len(data_file_positive)) / float(len(data_file_negative))
        return KeelData(file_name, rows, features, imbalance_ratio, x, y, xy)

    def read_keel_dat_directory(self):
        return list(self.keel_dat_file_dir_generator())

    def keel_dat_file_dir_generator(self):
        files_wildcard = self.data_dir + '/*.dat'
        file_paths = glob.glob(files_wildcard)
        for file_path in file_paths:
            yield self.read_keel_dat_file(path.basename(file_path))
