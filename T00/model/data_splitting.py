import pandas as pd
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import statistics
from sklearn.model_selection import GroupShuffleSplit

class DataSplitting(object):

    def __init__(self, 
                 cleaning_data, 
                 random_seed=42):

        self.processed = cleaning_data

    def worker(self):
        print("Data Splitting in progress...")

        splitter = GroupShuffleSplit(test_size=.2, n_splits=2, random_state = 7)
        split = splitter.split(self.processed, groups = self.processed.gene_id)
        train_inds, test_inds = next(split)
        train = self.processed.iloc[train_inds]
        test = self.processed.iloc[test_inds]
        xtrain = train[['new_dwell_0', 'new_stdv_0', 'new_mean_0', 'new_dwell_1', 'new_stdv_1', 'new_mean_1', 'new_dwell_2', 'new_stdv_2',
            'new_mean_2', 'n_1A', 'n_1C', 'n_1G', 'n_1T', 'n_2A', 'n_2C', 'n_2G',
            'n_2T', 'n_3A', 'n_3C', 'n_3G', 'n_3T', 'n_4A', 'n_4C', 'n_4G', 'n_4T',
            'n_5A', 'n_5C', 'n_5G', 'n_5T', 'n_6A', 'n_6C', 'n_6G', 'n_6T', 'n_7A',
            'n_7C', 'n_7G', 'n_7T', 'new_median_dwell_0', 'new_median_stdv_0', 'new_median_mean_0', 'new_median_dwell_1', 'new_median_stdv_1', 'new_median_mean_1', 'new_median_dwell_2', 'new_median_stdv_2', 'new_median_mean_2','new_transcript_id', 'position_id']]
        xtest = test[['new_dwell_0', 'new_stdv_0', 'new_mean_0', 'new_dwell_1', 'new_stdv_1', 'new_mean_1', 'new_dwell_2', 'new_stdv_2',
            'new_mean_2', 'n_1A', 'n_1C', 'n_1G', 'n_1T', 'n_2A', 'n_2C', 'n_2G',
            'n_2T', 'n_3A', 'n_3C', 'n_3G', 'n_3T', 'n_4A', 'n_4C', 'n_4G', 'n_4T',
            'n_5A', 'n_5C', 'n_5G', 'n_5T', 'n_6A', 'n_6C', 'n_6G', 'n_6T', 'n_7A',
            'n_7C', 'n_7G', 'n_7T', 'new_median_dwell_0', 'new_median_stdv_0', 'new_median_mean_0', 'new_median_dwell_1', 'new_median_stdv_1', 'new_median_mean_1', 'new_median_dwell_2', 'new_median_stdv_2', 'new_median_mean_2','new_transcript_id', 'position_id']]
        ytrain = train['label']
        ytest = test['label']

        print("Data split!")
        return xtest, xtrain, ytest, ytrain