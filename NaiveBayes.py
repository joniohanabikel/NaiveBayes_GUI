import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# Discretization
class NaiveBayesClassifier:
    def __init__(self, dir: "str", bins: "int"):
        """
        Preprocess, Build and Classify NaiveBayes model.
        :param dir: path to working directory
        :param bins: user choice of number of bins to discrete numeric values
        """
        self.working_directory = dir
        os.chdir(self.working_directory)
        self.struct = self.read_structure()
        self.bins = bins
        self.train_path = 'train.csv'

    def fillna(self, col, type):
        if type == 'OBJECT':
            col.fillna(col.mode().values[0], inplace=True)
            print("{} is string {}".format(col.name, col.dtype))
        else:
            col.fillna(col.mean(), inplace=True)
            print("{} is numeric {}".format(col.name, col.dtype))
        return col

    def read_structure(self):
        struct = pd.read_csv("Structure.txt", sep=" ", names=["attr", "name", "unique_values"], header=None)
        struct = struct.iloc[:, 1:3]
        return struct

    def discrete_numeric(self, col: pd.Series, num_of_bins, labels):
        pd.cut(col, bins=num_of_bins, labels=labels)
        return col

    def preprocess(self):
        encoder = LabelEncoder()
        df = pd.read_csv('train.csv')
        Y = df['class'].copy()
        df = df.drop('class', axis=1)
        data = {}
        #         data['features'] = df.columns.values
        #         print(data)
        #         data['class'] = Y.name
        for col in df.columns:
            features = {}
            data['features'] = []  # every feature can be accesed vie data['features'][feature_name]
            print(data)

            if self.struct.loc[self.struct.name == col, 'unique_values'].values[0] == 'NUMERIC':
                dtype = 'NUMERIC'
                data['features'][col]['type'] = 'dtype'
                data['features'][col]['rawdata'] = self.fillna(df[col], dtype)
                data['features'][col]['lables'] = [x for x in range(num_of_bins)]
                data['features'][col]['values'] = self.discrete_numeric(data['features'][col]['data'], self.bins,
                                                                        data['features'][col]['lables'])
            else:
                dtype = 'OBJECT'
                data['features'][col]['type'] = dtype
                data['features'][col]['rawdata'] = self.fillna(df[col], dtype)
                encoder.fit(data['features'][col]['rawdata'])
                data['features'][col]['lables'] = encoder.classes_
                data['features'][col]['values'] = encoder.transform(data['features'][col]['rawdata'])

        self.model = data

    def build_pipeline(self):
        self.preprocess()
        return "Building classifier using train-set is done!"
