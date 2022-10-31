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

class ModelTraining(object):

    def __init__(self, 
                 xtest, 
                 xtrain,
                 ytest,
                 ytrain, 
                 random_seed=42):

        self.xtest = xtest
        self.xtrain = xtrain
        self.ytest = ytest
        self.ytrain = ytrain

    def worker(self):

        print("training model")
        print("\n")
        model = XGBClassifier(min_child_weight=1, learning_rate=0.1, scale_pos_weight=18)
        model.fit(self.xtrain, self.ytrain)

        y_pred = model.predict(self.xtest)
        accuracy = accuracy_score(self.ytest, y_pred)
        precision, recall, thresholds = precision_recall_curve(self.ytest, y_pred)
        print("Accuracy of model :", accuracy)
        print("ROC AUC score :", roc_auc_score(self.ytest, y_pred))
        print("AUC :", auc(recall, precision))
        print("Classification :", classification_report(self.ytest, y_pred))

        y_pred_score = model.predict_proba(self.xtest)
        print("Y Pred score :", y_pred_score)
        cf_matrix = confusion_matrix(self.ytest, y_pred)
        print("\n")
        print(cf_matrix)
        sns.heatmap(cf_matrix, annot=True)
        #plt.show()

        return "group_t00 2022"