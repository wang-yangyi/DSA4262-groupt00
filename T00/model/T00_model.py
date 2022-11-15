import pandas as pd
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import statistics
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, auc, precision_recall_curve
import seaborn as sns
import time
from datetime import date
import seaborn as sns
import arviz as az
import warnings
import os

from .data_cleaning import DataCleaning
from .data_splitting import DataSplitting
from .model_training import ModelTraining
from .data_cleaning import SGnexCleaning

random_seed=42
np.random.seed(random_seed)
sns.set_style('whitegrid')
az.style.use("arviz-darkgrid")

class T00_model(object):

    print('Model Invoked')
    directory = str(os.getcwd())+ '/T00_model.joblib'
    print(directory)

    def __init__(self,random_seed=42):
        self.raw_data = []
        
    def train_model(self, raw_json, data_info):
        self.raw_json = raw_json
        self.data_info = data_info
        cleaning_data = DataCleaning(self.raw_json, self.data_info).data_cleaner()
        print("cleaning done")
        xtest, xtrain, ytest, ytrain  = DataSplitting(cleaning_data).worker()
        print("splitting done")
        return ModelTraining().worker(xtest, xtrain, ytest, ytrain)

    def test_SGnex_data(self, SGnex_data):
        warnings.filterwarnings('ignore')
        self.SGnex_data = SGnex_data
        print("data imported")
        data_x_1, data_x_2, processed = SGnexCleaning(self.SGnex_data).data_cleaner()
        print("data cleaned")
        return ModelTraining().SGnex_tester(data_x_1, data_x_2, processed)


