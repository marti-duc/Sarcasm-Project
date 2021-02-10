#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 12:50:29 2021

@author: Martina
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
pd.set_option('display.max_columns', None)


"""load data"""
csv_file = 'Desktop/liwc_response_context_train.csv'   
data = pd.read_csv(csv_file)
data = data.drop(['Unnamed: 0','Unnamed: 0.1.1.1','Unnamed: 0.1.1', 'Unnamed: 0.1'],axis = 1) #remove unwanted columns
#drop all categorical (for the time being)
featuresDF= data.copy()

featuresDF['response_emotion'] = featuresDF['response_emotion'].astype('category').cat.codes
featuresDF['context0_emotion'] = featuresDF['context0_emotion'].astype('category').cat.codes
featuresDF['context1_emotion'] = featuresDF['context1_emotion'].astype('category').cat.codes

featuresDF = featuresDF.drop(['tokenized_text','response','context/1','context/0','concat_tweet'], axis =1)

"""Encoding"""
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
data['label'] = labelencoder.fit_transform(data['label']) #numerically encode labels
labelDF = data['label']
features, labels = featuresDF.values, labelDF.values
headers = list(data.drop("label", axis=1))


feature_array = featuresDF.values
label_array = labelDF.values

X = feature_array
Y = label_array

featuresDF.head(2)  #df with all features extracted


"""Univariate"""

"""
Statistical tests can be used to select those features that have the strongest
relationship with the output variable. The scikit-learn library provides the 
SelectKBest class that can be used with a suite of different statistical tests 
to select a specific number of features.
"""

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
# UNIVARIATE FEATURE SELECTION
# drop target columns
drop_cols=['label']
X = featuresDF.drop(drop_cols, axis = 1) # X = independent columns (potential predictors)
y = featuresDF['label'] # y = target column (what we want to predict)
# instantiate SelectKBest to determine 20 best features
best_features = SelectKBest(score_func=f_classif, k=20)
fit = best_features.fit(X,y)
df_scores = pd.DataFrame(fit.scores_)
df_columns = pd.DataFrame(X.columns)
# concatenate dataframes
feature_scores = pd.concat([df_columns, df_scores],axis=1)
feature_scores.columns = ['Feature_Name','Score']  # name output columns
print(feature_scores.nlargest(20,'Score'))  # print 20 best features
# export selected features to .csv
df_univ_feat = feature_scores.nlargest(20,'Score')
df_univ_feat.to_csv('feature_selection_UNIVARIATE.csv', index=False)



"""Feature Importance"""


"""
You can get the feature importance of each feature of your dataset by
using the feature importance property of the model. Feature importance 
gives you a score for each feature of your data, the higher the score more 
important or relevant is the feature towards your output variable. Feature 
importance is an inbuilt class that comes with Tree Based Classifiers, 
we will be using Extra Tree Classifier for extracting the top 10 features for the dataset.
"""


# Feature importance is inbuilt with Tree Based Classifiers
from matplotlib import pyplot as plt

# FEATURE IMPORTANCE FEATURE SELECTION
np.random.seed(42)
# drop target columns

# instantiate RandomForestClassifier
rf_model = RandomForestClassifier()
rf_model.fit(X,Y)
feat_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
# determine 20 most important features
df_imp_feat = feat_importances.nlargest(20)
# print(rf_model.feature_importances_)
# export selected features to .csv
# df_imp_feat.to_csv('feature_selection_IMPORTANCE.csv', index=False)
df_imp_feat.to_csv('feature_selection_IMPORTANCE.csv')
# plot 20 most important features
# feat_importances.nlargest(20).plot(kind='barh')
df_imp_feat.plot(kind='barh')
plt.show()
print(df_imp_feat)