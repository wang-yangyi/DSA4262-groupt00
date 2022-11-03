import pandas as pd
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import statistics
import math

class DataCleaning(object):

    def __init__(self, 
                 raw_json, 
                 data_info,
                 random_seed=42):

        self.raw_json = raw_json
        self.data_info = data_info
        self.labels = self.import_info()
        self.imported_data = self.import_data()

    def extract_transcript_id(self, x):
        keys = x.keys()
        transcript = ''
        for i in keys:
            transcript = i
        return transcript

    def extract_transcript_position(self, id, raw):
        raw_dict = raw[id]
        keys = raw_dict.keys()
        position = ''
        for i in keys:
            position = i
        position = int(position)
        return position

    def extract_nucleotides(self, pos, id, raw):
        pos = str(pos)
        raw_dict = raw[id][pos]
        keys = raw_dict.keys()
        nucleo = ''
        for i in keys:
            nucleo = i
        return nucleo

    def extract_values(self, nucleo,pos,id, raw):
        pos = str(pos)
        values = raw[id][pos][nucleo]
        return values

    def extract_features(self, features):
        dwell_0 = []
        stdv_0 = []
        mean_0 = []
        dwell_1 = []
        stdv_1 = []
        mean_1 = []
        dwell_2 = []
        stdv_2 = []
        mean_2 = []
        
        for i in range(len(features)):
            dwell_0.append(features[i][0])
            stdv_0.append(features[i][1])
            mean_0.append(features[i][2])
            dwell_1.append(features[i][3])
            stdv_1.append(features[i][4])
            mean_1.append(features[i][5])
            dwell_2.append(features[i][6])
            stdv_2.append(features[i][7])
            mean_2.append(features[i][8])
        return dwell_0, stdv_0, mean_0, dwell_1, stdv_1, mean_1, dwell_2, stdv_2, mean_2

    def calculate_mean(self, x):
        avg = sum(x)/len(x)
        return avg

    def onehote(self, sequence):
        mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
        seq2 = [mapping[i] for i in sequence]
        return np.eye(4)[seq2]

    def split_nucleo(self, x):
        a = x[0]
        b = x[1]
        c = x[2]
        d = x[3]
        e = x[4]
        f = x[5]
        g = x[6]
        return a,b,c,d,e,f,g

    def import_data(self):
        print("Importing Datasets")
        f = open(self.raw_json, "r")
        raw = f.readlines()
        df = pd.DataFrame(raw, columns = ['raw'])
        print("Dataset imported")
        return df

    def import_info(self):
        print("Importing info data")
        labels = pd.read_csv(self.data_info, sep = ',', header=0)
        print("Info imported")
        return labels

    def data_cleaner(self):
        print('Data Cleaning in progress...')
        #reference self import data
        df = self.imported_data

        #clean columns
        df['raw'] = df.apply(lambda x: json.loads(x[0]), axis = 1)
        df['transcript_id'] = df.apply(lambda x: self.extract_transcript_id(x['raw']), axis = 1)
        df['position_id'] = df.apply(lambda x: self.extract_transcript_position(x['transcript_id'], x['raw']), axis = 1)
        df['nucleotides'] = df.apply(lambda x: self.extract_nucleotides(x['position_id'], x['transcript_id'], x['raw']), axis = 1)
        df['features'] = df.apply(lambda x: self.extract_values(x['nucleotides'], x['position_id'], x['transcript_id'], x['raw']), axis = 1)
        df[['dwell_0', 'stdv_0', 'mean_0', 'dwell_1', 'stdv_1', 'mean_1', 'dwell_2', 'stdv_2', 'mean_2']] = df.apply(lambda x: self.extract_features(x['features']), axis = 1, result_type = 'expand')

        #process and encode data
        labels = self.labels
        data = df.iloc[:,1:]
        merged = data.merge(labels, how = "left", left_on = ['transcript_id', 'position_id'], right_on = ['transcript_id', 'transcript_position'])
        merged = merged.drop('transcript_position', axis = 1)
        merged = merged.drop('features', axis = 1)
        processed = merged.copy()
        processed[['new_dwell_0', 'new_stdv_0', 'new_mean_0', 'new_dwell_1', 'new_stdv_1', 'new_mean_1', 'new_dwell_2', 'new_stdv_2', 'new_mean_2']] = processed[['dwell_0', 'stdv_0', 'mean_0', 'dwell_1', 'stdv_1', 'mean_1', 'dwell_2', 'stdv_2', 'mean_2']].applymap(self.calculate_mean)
        processed['new_nucleotides'] = processed['nucleotides'].apply(self.onehote)
        processed[["nucleotide_1","nucleotide_2","nucleotide_3","nucleotide_4","nucleotide_5","nucleotide_6","nucleotide_7"]] = processed.apply(lambda x: self.split_nucleo(x['new_nucleotides']), axis = 1, result_type = 'expand')
        processed[['n_1A','n_1C','n_1G','n_1T']] = pd.DataFrame(processed['nucleotide_1'].tolist())
        processed[['n_2A','n_2C','n_2G','n_2T']] = pd.DataFrame(processed['nucleotide_2'].tolist())
        processed[['n_3A','n_3C','n_3G','n_3T']] = pd.DataFrame(processed['nucleotide_3'].tolist())
        processed[['n_4A','n_4C','n_4G','n_4T']] = pd.DataFrame(processed['nucleotide_4'].tolist())
        processed[['n_5A','n_5C','n_5G','n_5T']] = pd.DataFrame(processed['nucleotide_5'].tolist())
        processed[['n_6A','n_6C','n_6G','n_6T']] = pd.DataFrame(processed['nucleotide_6'].tolist())
        processed[['n_7A','n_7C','n_7G','n_7T']] = pd.DataFrame(processed['nucleotide_7'].tolist())
        lb = LabelEncoder()
        enc = OneHotEncoder()
        processed['new_transcript_id'] = lb.fit_transform(processed['transcript_id'])
        processed[['new_median_dwell_0', 'new_median_stdv_0', 'new_median_mean_0', 'new_median_dwell_1', 'new_median_stdv_1', 'new_median_mean_1', 'new_median_dwell_2', 'new_median_stdv_2', 'new_median_mean_2']] = processed[['dwell_0', 'stdv_0', 'mean_0', 'dwell_1', 'stdv_1', 'mean_1', 'dwell_2', 'stdv_2', 'mean_2']].applymap(statistics.median)
        processed[['new_dwell_0', 'new_stdv_0', 'new_mean_0', 'new_dwell_1', 'new_stdv_1', 'new_mean_1', 'new_dwell_2', 'new_stdv_2', 'new_mean_2']] = processed[['new_dwell_0', 'new_stdv_0', 'new_mean_0', 'new_dwell_1', 'new_stdv_1', 'new_mean_1', 'new_dwell_2', 'new_stdv_2', 'new_mean_2']].transform(np.log)
        print('Data Processed')


        return processed

class SGnexCleaning(object):

    def __init__(self, 
                 raw_json, 
                 random_seed=42):

        self.raw_json = raw_json
        self.imported_data = self.import_data()

    def extract_transcript_id(self, x):
        keys = x.keys()
        transcript = ''
        for i in keys:
            transcript = i
        return transcript

    def extract_transcript_position(self, id, raw):
        raw_dict = raw[id]
        keys = raw_dict.keys()
        position = ''
        for i in keys:
            position = i
        position = int(position)
        return position

    def extract_nucleotides(self, pos, id, raw):
        pos = str(pos)
        raw_dict = raw[id][pos]
        keys = raw_dict.keys()
        nucleo = ''
        for i in keys:
            nucleo = i
        return nucleo

    def extract_values(self, nucleo,pos,id, raw):
        pos = str(pos)
        values = raw[id][pos][nucleo]
        return values

    def extract_features(self, features):
        dwell_0 = []
        stdv_0 = []
        mean_0 = []
        dwell_1 = []
        stdv_1 = []
        mean_1 = []
        dwell_2 = []
        stdv_2 = []
        mean_2 = []
        
        for i in range(len(features)):
            dwell_0.append(features[i][0])
            stdv_0.append(features[i][1])
            mean_0.append(features[i][2])
            dwell_1.append(features[i][3])
            stdv_1.append(features[i][4])
            mean_1.append(features[i][5])
            dwell_2.append(features[i][6])
            stdv_2.append(features[i][7])
            mean_2.append(features[i][8])
        return dwell_0, stdv_0, mean_0, dwell_1, stdv_1, mean_1, dwell_2, stdv_2, mean_2

    def calculate_mean(self, x):
        avg = sum(x)/len(x)
        return avg

    def onehote(self, sequence):
        mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
        seq2 = [mapping[i] for i in sequence]
        return np.eye(4)[seq2]

    def split_nucleo(self, x):
        a = x[0]
        b = x[1]
        c = x[2]
        d = x[3]
        e = x[4]
        f = x[5]
        g = x[6]
        return a,b,c,d,e,f,g

    def import_data(self):
        print("Importing Datasets")
        f = open(self.raw_json, "r")
        raw = f.readlines()
        df = pd.DataFrame(raw, columns = ['raw'])
        print("Dataset imported")
        return df

    def data_cleaner(self):
        print('Data Cleaning in progress...')
        df1 = self.imported_data

        df1['raw'] = df1.apply(lambda x: json.loads(x[0]), axis = 1)
        df1['transcript_id'] = df1.apply(lambda x: self.extract_transcript_id(x['raw']), axis = 1)
        df1['position_id'] = df1.apply(lambda x: self.extract_transcript_position(x['transcript_id'], x['raw']), axis = 1)
        df1['nucleotides'] = df1.apply(lambda x: self.extract_nucleotides(x['position_id'], x['transcript_id'], x['raw']), axis = 1)
        df1['features'] = df1.apply(lambda x: self.extract_values(x['nucleotides'], x['position_id'], x['transcript_id'], x['raw']), axis = 1)
        df1[['dwell_0', 'stdv_0', 'mean_0', 'dwell_1', 'stdv_1', 'mean_1', 'dwell_2', 'stdv_2', 'mean_2']] = df1.apply(lambda x: self.extract_features(x['features']), axis = 1, result_type = 'expand')
        processed = df1.copy()

        processed_1 = processed[0:math.floor(0.5*len(processed))]
        processed_2 = processed[math.floor(0.5*len(processed)):len(processed)]

        processed_1[['new_dwell_0', 'new_stdv_0', 'new_mean_0', 'new_dwell_1', 'new_stdv_1', 'new_mean_1', 'new_dwell_2', 'new_stdv_2', 'new_mean_2']] = processed_1[['dwell_0', 'stdv_0', 'mean_0', 'dwell_1', 'stdv_1', 'mean_1', 'dwell_2', 'stdv_2', 'mean_2']].applymap(self.calculate_mean)
        processed_1['new_nucleotides'] = processed_1['nucleotides'].apply(self.onehote)
        processed_1[["nucleotide_1","nucleotide_2","nucleotide_3","nucleotide_4","nucleotide_5","nucleotide_6","nucleotide_7"]] = processed_1.apply(lambda x: self.split_nucleo(x['new_nucleotides']), axis = 1, result_type = 'expand')
        processed_1[['n_1A','n_1C','n_1G','n_1T']] = pd.DataFrame(processed_1['nucleotide_1'].tolist())
        processed_1[['n_2A','n_2C','n_2G','n_2T']] = pd.DataFrame(processed_1['nucleotide_2'].tolist())
        processed_1[['n_3A','n_3C','n_3G','n_3T']] = pd.DataFrame(processed_1['nucleotide_3'].tolist())
        processed_1[['n_4A','n_4C','n_4G','n_4T']] = pd.DataFrame(processed_1['nucleotide_4'].tolist())
        processed_1[['n_5A','n_5C','n_5G','n_5T']] = pd.DataFrame(processed_1['nucleotide_5'].tolist())
        processed_1[['n_6A','n_6C','n_6G','n_6T']] = pd.DataFrame(processed_1['nucleotide_6'].tolist())
        processed_1[['n_7A','n_7C','n_7G','n_7T']] = pd.DataFrame(processed_1['nucleotide_7'].tolist())

        processed_2[['new_dwell_0', 'new_stdv_0', 'new_mean_0', 'new_dwell_1', 'new_stdv_1', 'new_mean_1', 'new_dwell_2', 'new_stdv_2', 'new_mean_2']] = processed_1[['dwell_0', 'stdv_0', 'mean_0', 'dwell_1', 'stdv_1', 'mean_1', 'dwell_2', 'stdv_2', 'mean_2']].applymap(self.calculate_mean)
        processed_2['new_nucleotides'] = processed_2['nucleotides'].apply(self.onehote)
        processed_2[["nucleotide_1","nucleotide_2","nucleotide_3","nucleotide_4","nucleotide_5","nucleotide_6","nucleotide_7"]] = processed_2.apply(lambda x: self.split_nucleo(x['new_nucleotides']), axis = 1, result_type = 'expand')
        processed_2[['n_1A','n_1C','n_1G','n_1T']] = pd.DataFrame(processed_2['nucleotide_1'].tolist())
        processed_2[['n_2A','n_2C','n_2G','n_2T']] = pd.DataFrame(processed_2['nucleotide_2'].tolist())
        processed_2[['n_3A','n_3C','n_3G','n_3T']] = pd.DataFrame(processed_2['nucleotide_3'].tolist())
        processed_2[['n_4A','n_4C','n_4G','n_4T']] = pd.DataFrame(processed_2['nucleotide_4'].tolist())
        processed_2[['n_5A','n_5C','n_5G','n_5T']] = pd.DataFrame(processed_2['nucleotide_5'].tolist())
        processed_2[['n_6A','n_6C','n_6G','n_6T']] = pd.DataFrame(processed_2['nucleotide_6'].tolist())
        processed_2[['n_7A','n_7C','n_7G','n_7T']] = pd.DataFrame(processed_2['nucleotide_7'].tolist())

        lb = LabelEncoder()
        processed_1['new_transcript_id'] = lb.fit_transform(processed_1['transcript_id'])
        processed_1[['new_median_dwell_0', 'new_median_stdv_0', 'new_median_mean_0', 'new_median_dwell_1', 'new_median_stdv_1', 'new_median_mean_1', 'new_median_dwell_2', 'new_median_stdv_2', 'new_median_mean_2']] = processed_1[['dwell_0', 'stdv_0', 'mean_0', 'dwell_1', 'stdv_1', 'mean_1', 'dwell_2', 'stdv_2', 'mean_2']].applymap(statistics.median)
        processed_1[['new_dwell_0', 'new_stdv_0', 'new_mean_0', 'new_dwell_1', 'new_stdv_1', 'new_mean_1', 'new_dwell_2', 'new_stdv_2', 'new_mean_2']] = processed_1[['new_dwell_0', 'new_stdv_0', 'new_mean_0', 'new_dwell_1', 'new_stdv_1', 'new_mean_1', 'new_dwell_2', 'new_stdv_2', 'new_mean_2']].transform(np.log)
        processed_2['new_transcript_id'] = lb.fit_transform(processed_2['transcript_id'])
        processed_2[['new_median_dwell_0', 'new_median_stdv_0', 'new_median_mean_0', 'new_median_dwell_1', 'new_median_stdv_1', 'new_median_mean_1', 'new_median_dwell_2', 'new_median_stdv_2', 'new_median_mean_2']] = processed_2[['dwell_0', 'stdv_0', 'mean_0', 'dwell_1', 'stdv_1', 'mean_1', 'dwell_2', 'stdv_2', 'mean_2']].applymap(statistics.median)
        processed_2[['new_dwell_0', 'new_stdv_0', 'new_mean_0', 'new_dwell_1', 'new_stdv_1', 'new_mean_1', 'new_dwell_2', 'new_stdv_2', 'new_mean_2']] = processed_2[['new_dwell_0', 'new_stdv_0', 'new_mean_0', 'new_dwell_1', 'new_stdv_1', 'new_mean_1', 'new_dwell_2', 'new_stdv_2', 'new_mean_2']].transform(np.log)

        data_x_1 = processed_1[['new_dwell_0', 'new_stdv_0', 'new_mean_0', 'new_dwell_1', 'new_stdv_1', 'new_mean_1', 'new_dwell_2', 'new_stdv_2',
            'new_mean_2', 'n_1A', 'n_1C', 'n_1G', 'n_1T', 'n_2A', 'n_2C', 'n_2G',
            'n_2T', 'n_3A', 'n_3C', 'n_3G', 'n_3T', 'n_4A', 'n_4C', 'n_4G', 'n_4T',
            'n_5A', 'n_5C', 'n_5G', 'n_5T', 'n_6A', 'n_6C', 'n_6G', 'n_6T', 'n_7A',
            'n_7C', 'n_7G', 'n_7T', 'new_median_dwell_0', 'new_median_stdv_0', 'new_median_mean_0', 'new_median_dwell_1', 'new_median_stdv_1', 'new_median_mean_1', 'new_median_dwell_2', 'new_median_stdv_2', 'new_median_mean_2','new_transcript_id', 'position_id']]

        data_x_2 = processed_2[['new_dwell_0', 'new_stdv_0', 'new_mean_0', 'new_dwell_1', 'new_stdv_1', 'new_mean_1', 'new_dwell_2', 'new_stdv_2',
            'new_mean_2', 'n_1A', 'n_1C', 'n_1G', 'n_1T', 'n_2A', 'n_2C', 'n_2G',
            'n_2T', 'n_3A', 'n_3C', 'n_3G', 'n_3T', 'n_4A', 'n_4C', 'n_4G', 'n_4T',
            'n_5A', 'n_5C', 'n_5G', 'n_5T', 'n_6A', 'n_6C', 'n_6G', 'n_6T', 'n_7A',
            'n_7C', 'n_7G', 'n_7T', 'new_median_dwell_0', 'new_median_stdv_0', 'new_median_mean_0', 'new_median_dwell_1', 'new_median_stdv_1', 'new_median_mean_1', 'new_median_dwell_2', 'new_median_stdv_2', 'new_median_mean_2','new_transcript_id', 'position_id']]

        return data_x_1, data_x_2, processed

