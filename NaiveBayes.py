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

    @staticmethod
    def calculate_prior(data):
        unique, count = np.unique(data, return_counts=True, axis=0)
        return count / len(pair)

    @staticmethod
    def calculate_likelihood(feature, target, m=2):
        feature_unique = np.unique(feature, return_counts=False, axis=0)
        # p = uniform prior (1/M, M – number of diff. values)
        print(feature_unique)
        p = 1 / len(feature_unique)
        print(f"p: = {p}")
        target_unique = np.unique(target, return_counts=False, axis=0)
        print(target_unique)
        data = pd.DataFrame(zip(feature, target), columns=['feature', 'target'])
        print(data)
        prob_df = pd.DataFrame(columns=['feature', 'target', 'likelihood'])
        for i in target_unique:
            # n : number of sample class value = i
            n = len(data.loc[data['target'] == i,])
            for j in feature_unique:
                # n_c: number of examples for which v(feature) = vj and c = ci
                n_c = len(data.loc[(data['target'] == i) & (data['feature'] == j),])
                # prob = (n_c + m*p) / (n +m )
                prob = (n_c + m * p) / (n + m)
                print("target = {}\nfeature = {}\nprob = {}".format(i, j, prob))
                prob_df = prob_df.append({'feature': j, 'target': i, 'likelihood': prob}, ignore_index=True)
        print(prob_df)
        return prob_df

    def preprocess(self):
        encoder = LabelEncoder()
        df = pd.read_csv('train.csv')
        Y = df['class'].copy()
        df = df.drop('class', axis=1)
        data = {}
        features = {}
        for column in df.columns:
            col = {}
            if self.struct.loc[self.struct.name == column, 'unique_values'].values[0] == 'NUMERIC':
                dtype = 'NUMERIC'
                col['type'] = 'dtype'
                col['rawdata'] = self.fillna(df[column], dtype)
                col['lables'] = [x for x in range(self.bins)]
                col['values'] = self.discrete_numeric(col['rawdata'], self.bins,
                                                      col['lables'])
            else:
                dtype = 'OBJECT'
                col['type'] = dtype
                col['rawdata'] = self.fillna(df[column], dtype)
                encoder.fit(col['rawdata'])
                col['lables'] = encoder.classes_
                col['values'] = encoder.transform(col['rawdata'])
            features[column] = col
        data['features'] = features

        # construct class
        dtype = 'OBJECT'
        data['class'] = {'dtype': dtype,
                         'rawdata': self.fillna(Y, dtype),
                         'lables': encoder.fit(Y).classes_,
                         'values': encoder.fit_transform(Y),
                         'prior': NaiveBayesClassifier.calculate_prior(encoder.fit_transform(Y)),
                         }
        self.model = data

    def build_pipeline(self):
        self.preprocess()
        #         self.calculate_probabilites()
        return "Building classifier using train-set is done!"










