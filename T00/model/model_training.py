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
from joblib import dump, load
import pickle
import os

class ModelTraining(object):

    def worker(self, xtest, xtrain, ytest, ytrain):

        print("training model")
        print("\n")
        model = XGBClassifier(min_child_weight=1, learning_rate=0.1, scale_pos_weight=18)
        model.fit(xtrain, ytrain)
        self.model_trained = model

        y_pred = model.predict(xtest)
        accuracy = accuracy_score(ytest, y_pred)
        precision, recall, thresholds = precision_recall_curve(ytest, y_pred)
        print("Accuracy of model :", accuracy)
        print("ROC AUC score :", roc_auc_score(ytest, y_pred))
        print("AUC :", auc(recall, precision))
        print("Classification :", classification_report(ytest, y_pred))

        y_pred_score = model.predict_proba(xtest)
        print("Y Pred score :", y_pred_score)
        cf_matrix = confusion_matrix(ytest, y_pred)
        print("\n")
        print(cf_matrix)
        sns.heatmap(cf_matrix, annot=True)
        
        #use to output saved model
        #dump(model, '/Users/nigel/Desktop/SCHOOLWORK/Y4S1/DSA4262/dataset/T00_model.joblib') 

        return "Model trained, written by: group T00"

    def SGnex_tester(self, data_x_1, data_x_2, processed):
        directory = str(os.getcwd())+ '/T00/model/T00_model.joblib'
        print('Testing model')

        xgb_model_loaded = pickle.load(open(directory, "rb"))
        print("Predicting scores")
        data_y_pred_score_1 = xgb_model_loaded.predict_proba(data_x_1)
        score_df_1 = pd.DataFrame(data_y_pred_score_1)
        data_y_pred_score_2 = xgb_model_loaded.predict_proba(data_x_2)
        score_df_2 = pd.DataFrame(data_y_pred_score_2)
        score_df = pd.concat([score_df_1, score_df_2], axis = 0)
        score_df = score_df.reset_index(drop = True)

        predicted_df = pd.concat([processed, score_df], axis = 1)
        predicted_df = predicted_df[['transcript_id', 'position_id', 1]]
        predicted_df.columns = ['transcript_id', 'transcript_position', 'score']
        print("\n")
        print("Scores generated")
        predicted_df.to_csv("groupt00_dataset2_1.csv", index = False)
        return 'SGnex outputs saved'
