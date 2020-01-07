import glob
from os import path

import numpy as np
import pandas as pd

from models.keel_data import KeelData


class DataReader:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def read_keel_dat_file(self, file_name: str, skip=13):
        file_path = self.data_dir + "/" + file_name
        self.remove_attribute_file_lines(file_path)
        data_file = pd.read_csv(file_path, skiprows=skip, header=None)
        data_file = data_file.rename(columns={data_file.columns[len(list(data_file)) - 1]: 'Class'})
        xy = data_file.values
        y = (xy[:, xy.shape[1] - 1]).astype(np.str)
        for idx, entry in enumerate(y):
            y[idx] = entry.strip()  # Trim whitespaces
        x = np.delete(xy, xy.shape[1] - 1, axis=1).astype(np.float32)
        rows = len(data_file)
        features = (len(list(data_file)) - 1)
        classes = len(np.unique(y))
        return KeelData(file_name, rows, features, classes, x, y)

    def remove_attribute_file_lines(self, file_name):
        with open(file_name, "r") as input_file:
            input_lines = input_file.readlines()
        with open(file_name, "w") as output_file:
            for line in input_lines:
                if not line.startswith('@'):
                    output_file.write(line)

    def read_keel_dat_directory(self):
        return list(self.keel_dat_file_dir_generator())

    def keel_dat_file_dir_generator(self):
        files_wildcard = self.data_dir + '/*.dat'
        file_paths = glob.glob(files_wildcard)
        for file_path in file_paths:
            yield self.read_keel_dat_file(path.basename(file_path))
