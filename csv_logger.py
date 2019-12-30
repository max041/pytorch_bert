import os
import shutil
from utils import Singleton
import pandas as pd
import numpy as np


class Logger(metaclass=Singleton):
    def __init__(self, dump_dir=None):
        self.dump_dir = dump_dir
        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)

        self.logs_dict = {}

    def log_all(self):
        for name, log in self.logs_dict.items():
            pd.DataFrame(data=log.data, columns=log.columns).to_csv(
                os.path.join(self.dump_dir, log.name), mode='a', index=False, header=False
            )
            log.clean()

    def add_log(self, name, columns):
        self.logs_dict[name] = Log(name, columns)
        pd.DataFrame(columns=columns).to_csv(os.path.join(self.dump_dir, name), index=False)

    def __getitem__(self, key):
        return self.logs_dict[key]


class Log:
    def __init__(self, name, columns):
        self.name = name
        self.columns = columns
        self.data_dict = {}
        for column in columns:
            self.data_dict[column] = []

    def __getitem__(self, key):
        return self.data_dict[key]

    @property
    def data(self):
        data = []
        for column in self.columns:
            data.append(self.data_dict[column])

        return np.array(data).transpose()

    def clean(self):
        for column in self.columns:
            del self.data_dict[column][:]