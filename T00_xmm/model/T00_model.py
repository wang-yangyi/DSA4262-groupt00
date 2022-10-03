from logging import raiseExceptions
from signal import raise_signal
from turtle import hideturtle
import pandas as pd
import numpy as np
import json
import time
from datetime import date
import seaborn as sns
import arviz as az

random_seed=42
np.random.seed(random_seed)
sns.set_style('whitegrid')
az.style.use("arviz-darkgrid")

class T00_model(object):

    print('Model Invoked')

    def __init__(self, 
                 raw_json, 
                 random_seed=42):

        self.raw_json = raw_json
        self.dtypes_dict = {
            'feature1': 'int64',
            'feature2': 'category',
            'feature3': 'float16',
            'feature4': 'int8'
            }

    def json_load_data(self):

        time_now = time.time()
        data = []
        with open(self.raw_json) as f:   
            for line in f:
                data.append(json.loads(line))

        total_time = time.time() - time_now
        print("Time taken: ")
        print(total_time // 60 , " min ", round(total_time % 60,1) ," sec")
        print("/n")
        return data

    def pandas_load_data(self):

        time_now = time.time()
        ## INSERT PANDAS HERE
        print("/n")
        return df
