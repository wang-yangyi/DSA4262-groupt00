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

def extract_transcript_id(x):
    keys = x.keys()
    transcript = ''
    for i in keys:
        transcript = i
    return transcript

def extract_transcript_position(id, raw):
    raw_dict = raw[id]
    keys = raw_dict.keys()
    position = ''
    for i in keys:
        position = i
    position = int(position)
    return position

def extract_nucleotides(pos, id, raw):
    pos = str(pos)
    raw_dict = raw[id][pos]
    keys = raw_dict.keys()
    nucleo = ''
    for i in keys:
        nucleo = i
    return nucleo

def extract_values(nucleo,pos,id, raw):
    pos = str(pos)
    values = raw[id][pos][nucleo]
    return values

def extract_features(features):
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

def calculate_mean(x):
    avg = sum(x)/len(x)
    return avg

def onehote(sequence):
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    seq2 = [mapping[i] for i in sequence]
    return np.eye(4)[seq2]

def split_nucleo(x):
    a = x[0]
    b = x[1]
    c = x[2]
    d = x[3]
    e = x[4]
    f = x[5]
    g = x[6]
    return a,b,c,d,e,f,g

f = open("data.json", "r")
raw = f.readlines()
df = pd.DataFrame(raw, columns = ['raw'])
df['raw'] = df.apply(lambda x: json.loads(x[0]), axis = 1)
df['transcript_id'] = df.apply(lambda x: extract_transcript_id(x['raw']), axis = 1)
df['position_id'] = df.apply(lambda x: extract_transcript_position(x['transcript_id'], x['raw']), axis = 1)
df['nucleotides'] = df.apply(lambda x: extract_nucleotides(x['position_id'], x['transcript_id'], x['raw']), axis = 1)
df['features'] = df.apply(lambda x: extract_values(x['nucleotides'], x['position_id'], x['transcript_id'], x['raw']), axis = 1)
df[['dwell_0', 'stdv_0', 'mean_0', 'dwell_1', 'stdv_1', 'mean_1', 'dwell_2', 'stdv_2', 'mean_2']] = df.apply(lambda x: extract_features(x['features']), axis = 1, result_type = 'expand')

labels = pd.read_csv('data.info', sep = ',', header=0)
data = df.iloc[:,1:]
merged = data.merge(labels, how = "left", left_on = ['transcript_id', 'position_id'], right_on = ['transcript_id', 'transcript_position'])
merged = merged.drop('transcript_position', axis = 1)
merged = merged.drop('features', axis = 1)
processed = merged.copy()
processed[['new_dwell_0', 'new_stdv_0', 'new_mean_0', 'new_dwell_1', 'new_stdv_1', 'new_mean_1', 'new_dwell_2', 'new_stdv_2', 'new_mean_2']] = processed[['dwell_0', 'stdv_0', 'mean_0', 'dwell_1', 'stdv_1', 'mean_1', 'dwell_2', 'stdv_2', 'mean_2']].applymap(calculate_mean)
processed['new_nucleotides'] = processed['nucleotides'].apply(onehote)
processed[["nucleotide_1","nucleotide_2","nucleotide_3","nucleotide_4","nucleotide_5","nucleotide_6","nucleotide_7"]] = processed.apply(lambda x: split_nucleo(x['new_nucleotides']), axis = 1, result_type = 'expand')
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
splitter = GroupShuffleSplit(test_size=.2, n_splits=2, random_state = 7)
split = splitter.split(processed, groups = processed['gene_id'])
train_inds, test_inds = next(split)
train = processed.iloc[train_inds]
test = processed.iloc[test_inds]
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
model = XGBClassifier(min_child_weight=1, learning_rate=0.1, scale_pos_weight=18)
model.fit(xtrain, ytrain)

y_pred = model.predict(xtest)
accuracy = accuracy_score(ytest, y_pred)
precision, recall, thresholds = precision_recall_curve(ytest, y_pred)
print(accuracy)
print(roc_auc_score(ytest, y_pred))
print(auc(recall, precision))
print(classification_report(ytest, y_pred))

y_pred_score = model.predict_proba(xtest)
cf_matrix = confusion_matrix(ytest, y_pred)
sns.heatmap(cf_matrix, annot=True)