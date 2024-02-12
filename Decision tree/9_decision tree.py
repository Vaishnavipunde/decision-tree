# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 09:17:30 2024

@author: rajendra
"""

import pandas as pd
df=pd.read_csv()
df.head()
inputs=df.drop('salary_more_than_100k',axis=1)
target=df['salary_more_than_100k']
from sklearn.preprocessing import LabelEncoder
le_company=LabelEncoder()
le_job=LabelEncoder()
le_degree=LabelEncoder()
inputs['company_n']=le_company.fit_transform(input('company'))

inputs['job_n']=le_job.fit_transform(input('job'))

inputs['degree_n']=le_degree.fit_transform(input('degree'))

input_n=inputs.drop(['company','job','degree'],axis='columns')
target

from sklearn import tree
model=tree.DecisionTreeClasssifier()
model.fit(input_n,target)
model.predict([[2,1,0]])
model.predict([[2,1,1]])












