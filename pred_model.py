# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 19:41:50 2021 (Classification model)

@author: Sreekanth Putsala
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv("C:\\Users\\home\\Desktop\\MLFLOW_FORECAST_PROJECT\\df.csv")

print(df['Type_of_Treatment'].value_counts(normalize=True))
print(df['Cost_of_Treatment'].value_counts(normalize=True))
#dropping the columns Patient_Unique ID, Appointment_unique ID, Unnamed:0 & Unnamed:0.1

df = df.drop(['Appointment_Unique_ID'],axis = 1)
df = df.drop(['Unnamed: 0'],axis = 1)
df = df.drop(['Unnamed: 0.1'],axis = 1)
df = df.drop(['Patient_Unique_ID'],axis = 1)

# BINNING - AGE, DISTANCE

### AGE ###

df.loc[(df['Age']>=3) & (df['Age']<=10), 'age'] = '3-10'
df.loc[(df['Age']>10) & (df['Age']<=30), 'age'] = '11-30'
df.loc[(df['Age']>30) & (df['Age']<=60), 'age'] = '31-60'
df.loc[(df['Age']>60), 'age'] = '60 plus'

### DISTANCE ###

df.loc[(df['Distance']<=4), 'distance'] = '1-4'
df.loc[(df['Distance']>4) & (df['Distance']<=7), 'distance'] = '5-7'
df.loc[(df['Distance']>=8) & (df['Distance']<=12), 'distance'] = '8-12'

### MAPPING
df[['Employment_status']] = df[['Employment_status']].replace(to_replace = {'Employed':1,'Unemployed':0})
df[['Insurance']] = df[['Insurance']].replace(to_replace = {'Yes':1,'No':0})
df[['Gender']] = df[['Gender']].replace(to_replace = {'F':0,'M':1})
df['Type_of_Treatment'] = df['Type_of_Treatment'].replace(to_replace = {'Root Canal':0,'Flap Surgery':1,'Extractions':2,'Braces':3,'Dental Implant': 4,'Restorations': 5,'Fixed Prosthetic Denture': 6,'Scaling': 7,'Pulpectomy':8})
df['Appointment Day'] = df['Appointment Day'].replace(to_replace = {'Weekend':0,'Weekday':1,'Holiday':2})
df['Cost_of_Treatment'] = df['Cost_of_Treatment'].replace(to_replace = {'1000-2000':0,'1500-2500':1,'1000-3000':2,'4000-6000':3,'6000-10000': 4,'5000-15000': 5,'25000-35000': 6,'30000-40000': 7})
df[['age']] = df[['age']].replace(to_replace = {'3-10':0,'11-30':1,'31-60':2, '60 plus':3})
df[['distance']] = df[['distance']].replace(to_replace = {'1-4':0,'5-7':1,'8-12':2})
#df[['Show up']] = df[['Show up']].replace(to_replace = {'YES':1,'NO':0})

### DROPPING 'AGE' AND 'DISTANCE' COLUMNS ###

df = df.drop(['Age'],axis = 1)
df = df.drop(['Distance'],axis = 1)
df

### RENAME 'SHOW UP', 'APPOINTMENTDAY', 'AGE' AND 'DISTANCE' COLUMNS ###

df = df.rename(columns={'Show up':'Show_up', 'Appointment Day':'Appointment_Day','age':'Age','distance':'Distance' })

#converting data type to categorical
df['Employment_status'] = df['Employment_status'].astype('category')
df['Insurance'] = df['Insurance'].astype('category')
df['Gender'] = df['Gender'].astype('category')
df['Type_of_Treatment'] = df['Type_of_Treatment'].astype('category')
df['Cost_of_Treatment'] = df['Cost_of_Treatment'].astype('category')
df['Appointment_Day'] = df['Appointment_Day'].astype('category')
df['Age'] = df['Age'].astype('category')
df['Distance'] = df['Distance'].astype('category')
#df['Show_up'] = df['Show_up'].astype('category')

"""#Model Building for prediction

#Splitting the data into training and testing sets.
"""

X = df.drop(['Show_up'], axis=1)
y = df['Show_up']

X_train, X_test, y_train, y_test = train_test_split(X, y)

print(y_train.value_counts(normalize=True))
print(y_test.value_counts(normalize=True))

Model = []
Test_Accuracy = []
Train_Accuracy = []

print(df.info())

"""#Decision Tree-Classifier Model"""

from sklearn.tree import DecisionTreeClassifier as DT

DTC_model = DT(criterion = 'entropy')
DTC_model.fit(X_train, y_train)

# Prediction on Test Data and Accuracy
preds = DTC_model.predict(X_test)
print("Crosstable_Test:")
print(pd.crosstab(y_test, preds, rownames=['Actual'], colnames=['Predictions']))
print("Test_Accuracy_model:")
print(np.mean(preds == y_test))
DTC_TEST = (np.mean(preds == y_test))
print("Classification_Report_Test:")
print(classification_report(y_test, DTC_model.predict(X_test)))
 
# Prediction on Train Data and Accuracy
preds = DTC_model.predict(X_train)
print("Crosstable_Train:")
print(pd.crosstab(y_train, preds, rownames = ['Actual'], colnames = ['Predictions']))
print("Train_Accuracy_model:")
print(np.mean(preds == y_train))
DTC_TRAIN = (np.mean(preds == y_train))
print("Classification_Report_Train:")
print(classification_report(y_train, DTC_model.predict(X_train)))

Model.append('DTC_model')
Test_Accuracy.append(DTC_TEST)
Train_Accuracy.append(DTC_TRAIN)

"""#Random Forest Classifier Model

**Reading classification report:**

1. Precision: Accuracy of positive predictions. Precision = TP/(TP + FP).
2. Recall: Fraction of positives that were correctly identified. Recall = TP/(TP+FN)
3. F1 Score = 2*(Recall * Precision) / (Recall + Precision) Best score is 1.0 and the worst is 0.0 (F1 score is what percent of positive predictions were correct?).
4. Support: Support is the number of actual occurrences of the class in the specified dataset.

Imbalanced support in the training data may indicate structural weaknesses, which may require stratified sampling or rebalancing. 
"""

from sklearn.ensemble import RandomForestClassifier

RFC_model = RandomForestClassifier(n_estimators=300)
RFC_model.fit(X_train, y_train)

#Prediction on Test Data and Accuracy
print("Test_Accuracy_RFC_model:")
print(RFC_model.score(X_test, y_test))
RFC_TEST = (RFC_model.score(X_test, y_test))
print("Confusion_Matrix_Test:")
print(confusion_matrix(y_test, RFC_model.predict(X_test)))
print("Classification_Report_Test:")
print(classification_report(y_test, RFC_model.predict(X_test)))

#Prediction on Train Data and Accuracy
print("Train_Accuracy_RFC_model:")
print(RFC_model.score(X_train, y_train))
RFC_TRAIN = (RFC_model.score(X_train, y_train))
print("Confusion_Matrix_Train:")
print(confusion_matrix(y_train, RFC_model.predict(X_train)))
print("Classification_Report_Train:")
print(classification_report(y_train, RFC_model.predict(X_train)))

Model.append('RFC_model')
Test_Accuracy.append(RFC_TEST)
Train_Accuracy.append(RFC_TRAIN)

"""#Plotting feature importance"""

feat_importances = pd.Series(RFC_model.feature_importances_, index=X.columns)
feat_importances = feat_importances.nlargest(20)
feat_importances.plot(kind='barh')
plt.show()

"""Fine-tuning Random Forest Classifier Model using GridSearchCV"""

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix

RFC_model_grid = RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=42)
param_grid = {"max_features": [8], "min_samples_split": [2]}
RFC_model_grid_search = GridSearchCV(RFC_model_grid, param_grid, n_jobs = -1, cv = 5, scoring = 'accuracy')
RFC_model_grid_search.fit(X_train, y_train)

print("Best_params:")
print(RFC_model_grid_search.best_params_)

cv_RFC_model_grid = RFC_model_grid_search.best_estimator_

#Prediction on Test Data and Accuracy
print("confusion_matrix_Test:")
print(confusion_matrix(y_test, cv_RFC_model_grid.predict(X_test)))
print("Test_Accuracy_RFC_model_grid_search:")
print(accuracy_score(y_test, cv_RFC_model_grid.predict(X_test)))
RFC_GRID_TEST = (accuracy_score(y_test, cv_RFC_model_grid.predict(X_test))) 
print("Classification_Report_Test:")
print(classification_report(y_test, cv_RFC_model_grid.predict(X_test)))

#Prediction on Train Data and Accuracy
print("confusion_matrix_Train:")
print(confusion_matrix(y_train, cv_RFC_model_grid.predict(X_train)))
print("Train_Accuracy_RFC_model_grid_search:")
print(accuracy_score(y_train, cv_RFC_model_grid.predict(X_train)))
RFC_GRID_TRAIN = (accuracy_score(y_train, cv_RFC_model_grid.predict(X_train)))
print("Classification_Report_Train:")
print(classification_report(y_train, cv_RFC_model_grid.predict(X_train)))

Model.append('RFC_model_grid')
Test_Accuracy.append(RFC_GRID_TEST)
Train_Accuracy.append(RFC_GRID_TRAIN)

"""#Bagging Classifier Model"""

from sklearn import tree
from sklearn.ensemble import BaggingClassifier

clftree = tree.DecisionTreeClassifier()
Bag_Clf_model = BaggingClassifier(base_estimator = clftree, n_estimators = 500,
                            bootstrap = True, n_jobs = 1, random_state = 42)
Bag_Clf_model.fit(X_train, y_train)

#Prediction on Test Data and Accuracy
print("confusion_matrix_Test:")
print(confusion_matrix(y_test, Bag_Clf_model.predict(X_test)))
print("Test_Accuracy_Bag_Clf_model:")
print(accuracy_score(y_test, Bag_Clf_model.predict(X_test)))
BC_TEST = (accuracy_score(y_test, Bag_Clf_model.predict(X_test)))
print("Classification_Report_Test:")
print(classification_report(y_test, Bag_Clf_model.predict(X_test)))

#Prediction on Train Data and Accuracy
print("confusion_matrix_Train:")
print(confusion_matrix(y_train, Bag_Clf_model.predict(X_train)))
print("Train_Accuracy_Bag_Clf_model:")
print(accuracy_score(y_train, Bag_Clf_model.predict(X_train)))
BC_TRAIN = (accuracy_score(y_train, Bag_Clf_model.predict(X_train)))
print("Classification_Report_Train:")
print(classification_report(y_train, Bag_Clf_model.predict(X_train)))

Model.append('Bag_Clf_model')
Test_Accuracy.append(BC_TEST)
Train_Accuracy.append(BC_TRAIN)

"""#AdaBoost Classifier Model"""

from sklearn.ensemble import AdaBoostClassifier

AdaB_Clf_model = AdaBoostClassifier(learning_rate = 0.02, n_estimators = 5000)
AdaB_Clf_model.fit(X_train, y_train)

#Prediction on Test Data and Accuracy
print("confusion_matrix_Test:")
print(confusion_matrix(y_test, AdaB_Clf_model.predict(X_test)))
print("Test_Accuracy_AdaB_Clf_model:")
print(accuracy_score(y_test, AdaB_Clf_model.predict(X_test)))
AdaB_TEST = (accuracy_score(y_test, AdaB_Clf_model.predict(X_test)))
print("Classification_Report_Test:")
print(classification_report(y_test, AdaB_Clf_model.predict(X_test)))

#Prediction on Train Data and Accuracy
print("confusion_matrix_Train:")
print(confusion_matrix(y_train, AdaB_Clf_model.predict(X_train)))
print("Train_Accuracy_AdaB_Clf_model:")
print(accuracy_score(y_train, AdaB_Clf_model.predict(X_train)))
AdaB_TRAIN = (accuracy_score(y_train, AdaB_Clf_model.predict(X_train)))
print("Classification_Report_Train:")
print(classification_report(y_train, AdaB_Clf_model.predict(X_train)))

Model.append('AdaB_Clf_model')
Test_Accuracy.append(AdaB_TEST)
Train_Accuracy.append(AdaB_TRAIN)

"""#Gradient Boosting Classifier Model"""

from sklearn.ensemble import GradientBoostingClassifier

GBoost_Clf_model = GradientBoostingClassifier()
GBoost_Clf_model.fit(X_train, y_train)

#Prediction on Test Data and Accuracy
print("confusion_matrix_Test:")
print(confusion_matrix(y_test, GBoost_Clf_model.predict(X_test)))
print("Test_Accuracy_GBoost_Clf_model:")
print(accuracy_score(y_test, GBoost_Clf_model.predict(X_test)))
GBC_TEST = (accuracy_score(y_test, GBoost_Clf_model.predict(X_test)))
print("Classification_Report_Test:")
print(classification_report(y_test, GBoost_Clf_model.predict(X_test)))

#Prediction on Train Data and Accuracy
print("confusion_matrix_Train:")
print(confusion_matrix(y_train, GBoost_Clf_model.predict(X_train)))
print("Trian_Accuracy_GBoost_Clf_model:")
print(accuracy_score(y_train, GBoost_Clf_model.predict(X_train)))
GBC_TRAIN = (accuracy_score(y_train, GBoost_Clf_model.predict(X_train)))
print("Classification_Report_Train:")
print(classification_report(y_train, GBoost_Clf_model.predict(X_train)))

Model.append('GBoost_Clf_model')
Test_Accuracy.append(GBC_TEST)
Train_Accuracy.append(GBC_TRAIN)

"""Fine-tuning Gradient Boosting Classifier Model by chnaging Hyperparameter values"""

GBoost_Clf_model2 = GradientBoostingClassifier(learning_rate = 0.02, n_estimators = 1000, max_depth = 1)
GBoost_Clf_model2.fit(X_train, y_train)

#Prediction on Test Data and Accuracy
print("confusion_matrix_Test:")
print(confusion_matrix(y_test, GBoost_Clf_model2.predict(X_test)))
print("Test_Accuracy_GBoost_Clf_model2:")
print(accuracy_score(y_test, GBoost_Clf_model2.predict(X_test)))
GBC2_TEST = (accuracy_score(y_test, GBoost_Clf_model2.predict(X_test)))
print("Classification_Report_Test:")
print(classification_report(y_test, GBoost_Clf_model2.predict(X_test)))

#Prediction on Train Data and Accuracy
print("confusion_matrix_Train:")
print(confusion_matrix(y_test, GBoost_Clf_model2.predict(X_test)))
print("Train_Accuracy_GBoost_Clf_model2:")
print(accuracy_score(y_train, GBoost_Clf_model2.predict(X_train)))
GBC2_TRAIN = accuracy_score(y_train, GBoost_Clf_model2.predict(X_train))
print("Classification_Report_Train:")
print(classification_report(y_train, GBoost_Clf_model2.predict(X_train)))

Model.append('GBoost_Clf_model2')
Test_Accuracy.append(GBC2_TEST)
Train_Accuracy.append(GBC2_TRAIN)

"""#Multinomial Naive Bayes

* Multinomial Naive Bayes changing default alpha for laplace smoothing.
* If alpha = 0 then no smoothing is applied and the default alpha parameter is 1.
* The smoothing process mainly solves the emergence of zero probability problem in the dataset.
"""

from sklearn.naive_bayes import MultinomialNB as MB

Naive_Bayes_Classifier_model = MB()
Naive_Bayes_Classifier_model.fit(X_train, y_train)

#Prediction on Test Data and Accuracy
print("confusion_matrix_Test:")
print(confusion_matrix(y_test, Naive_Bayes_Classifier_model.predict(X_test)))
print("Test_Naive_Bayes_Classifier_model:")
print(accuracy_score(y_test, Naive_Bayes_Classifier_model.predict(X_test)))
MNB_TEST = (accuracy_score(y_test, Naive_Bayes_Classifier_model.predict(X_test)))
print("Classification_Report_Test:")
print(classification_report(y_test, Naive_Bayes_Classifier_model.predict(X_test)))

#Prediction on Train Data and Accuracy
print("confusion_matrix_Train:")
print(confusion_matrix(y_train, Naive_Bayes_Classifier_model.predict(X_train)))
print("Train_Naive_Bayes_Classifier_model:")
print(accuracy_score(y_train, Naive_Bayes_Classifier_model.predict(X_train)))
MNB_TRAIN = (accuracy_score(y_train, Naive_Bayes_Classifier_model.predict(X_train)))
print("Classification_Report_Train:")
print(classification_report(y_train, Naive_Bayes_Classifier_model.predict(X_train)))

Model.append('Naive_Bayes_Classifier_model')
Test_Accuracy.append(MNB_TEST)
Train_Accuracy.append(MNB_TRAIN)

"""Finetunning Multinomial Naive Bayes Model by applying laplace"""

Naive_Bayes_Classifier_Lap_model = MB(alpha = 3)
Naive_Bayes_Classifier_Lap_model.fit(X_train, y_train)

#Prediction on Test Data and Accuracy
print("confusion_matrix_Test:")
print(confusion_matrix(y_test, Naive_Bayes_Classifier_Lap_model.predict(X_test)))
print("Test_Naive_Bayes_Classifier_Lap_model:")
print(accuracy_score(y_test, Naive_Bayes_Classifier_Lap_model.predict(X_test)))
MNB2_TEST = (accuracy_score(y_test, Naive_Bayes_Classifier_Lap_model.predict(X_test)))
print("Classification_Report_Test:")
print(classification_report(y_test, Naive_Bayes_Classifier_Lap_model.predict(X_test)))

#Prediction on Train Data and Accuracy
print("confusion_matrix_Train:")
print(confusion_matrix(y_train, Naive_Bayes_Classifier_Lap_model.predict(X_train)))
print("Train_Naive_Bayes_Classifier_Lap_model:")
print(accuracy_score(y_train, Naive_Bayes_Classifier_Lap_model.predict(X_train)))
MNB2_TRAIN = (accuracy_score(y_train, Naive_Bayes_Classifier_Lap_model.predict(X_train)))
print("Classification_Report_Train:")
print(classification_report(y_train, Naive_Bayes_Classifier_Lap_model.predict(X_train)))

Model.append('Naive_Bayes_Classifier_Lap_model')
Test_Accuracy.append(MNB2_TEST)
Train_Accuracy.append(MNB2_TRAIN)

Models_Summary = pd.DataFrame([Model,Test_Accuracy,Train_Accuracy]).T
Models_Summary.columns = ['CLASSIFICATION_ModelS','Test_Accuracy','Train_Accuracy']

Models_Summary

import pickle
with open('Dental_model_GBoost_Clf_model2.pkl', 'wb') as file: pickle.dump(GBoost_Clf_model2, file)

