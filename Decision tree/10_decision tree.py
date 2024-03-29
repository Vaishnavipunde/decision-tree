# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 08:24:56 2024

@author: rajendra
"""

import pandas as pd
import numpy as np 
from sklearn.preprocessing import LabelEncoder

data=pd.read_csv("C:/2-dataset/credit.csv")
data.isnull().sum
data.dropna()
data.columns
data=data.drop(['phone'],axis=1)


lb=LabelEncoder()
data['checking_balance']=lb.fit_transform(data['checking_balance'])

data['credit_history']=lb.fit_transform(data['credit_history'])

data['purpose']=lb.fit_transform(data['purpose'])

data['savings_balance']=lb.fit_transform(data['savings_balance'])

data['employment_duration']=lb.fit_transform(data['employment_duration'])

data['other_credit']=lb.fit_transform(data['other_credit'])

data['housing']=lb.fit_transform(data['housing'])

data['job']=lb.fit_transform(data['job'])

data['default'].unique()
data['default'].value_counts()
colnames=list(data.columns)

predictors=colnames[:15]
target=colnames[:15]

from sklearn.model_selection import train_test_split
train,test=train_test_split(data, test_size=0.3)

from sklearn.tree import DecisionTreeClassifier as DT

model=DT(criterion='entropy')
model.fit(train[predictors],train[target])

#accuracy on test dataset
preds=model.predict(test[predictors])
preds
pd.crosstab(test[target], preds,rownames=['Actual'],colnames=['predictions'])
np.mean(preds==test[target])

#accuracy on train dataset
preds=model.predict(train[predictors])
preds
pd.crosstab(train[target], preds,rownames=['Actual'],colnames=['predictions'])
np.mean(preds==train[target])
#accuracy of train data is more than test so it is overfit model



































































