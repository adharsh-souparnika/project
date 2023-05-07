#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor
import pickle

import warnings
warnings.filterwarnings('ignore')


# In[2]:


data=pd.read_excel('WorldCupMatches1.xlsx')
data


# In[3]:



data.isna().sum()


# In[4]:


data=data.drop(['Datetime','City','Stadium'],axis=1)


# In[5]:


le1=LabelEncoder()
data['Stage']=le1.fit_transform(data['Stage'])
le2=LabelEncoder()
data['Home Team Name']=le2.fit_transform(data['Home Team Name'])
le3=LabelEncoder()
data['Away Team Name']=le3.fit_transform(data['Away Team Name'])


# In[6]:


data


# In[7]:


data


# In[8]:


x=data.drop('Attendance',axis=1)
y=data['Attendance']


# In[9]:


y


# In[10]:


model_params={'svm':{'model':SVR(gamma='auto'),'params':{'C':[10,15,20],'kernel':['linear','sigmoid','rbf']}},
             'random forest':{'model':RandomForestRegressor(),'params':{'n_estimators':[10,15,20]}},
             'decision tree':{'model':DecisionTreeRegressor(),'params':{'max_depth':[10,15,20]}},
             'linear regression':{'model':LinearRegression(),'params':{}},
              'adaboost':{'model':AdaBoostRegressor(),'params':{}},
              'xgboost':{'model':XGBRegressor(),'params':{}},
             'kneighbors':{'model':KNeighborsRegressor(),'params':{'n_neighbors':[10,15,20]}}}


# In[11]:


scores=[]
models=['svm','random forest','decision tree','kneighbors','adaboost','xgboost']
for model_name in models:
    mp=model_params[model_name]
    gds=GridSearchCV(mp['model'],mp['params'],cv=5,return_train_score=True)
    gds.fit(x,y)
    scores.append({'model':model_name,'best_score':gds.best_score_, 'best_params':gds.best_params_})


# In[12]:


scores


# In[13]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=25)


# In[14]:


model=RandomForestRegressor(n_estimators=47,random_state=3)
model.fit(x_train,y_train)
model.score(x_test,y_test)


# In[15]:


model1=LinearRegression()
model1.fit(x_train,y_train)
model1.score(x_test,y_test)


# In[16]:


model2=DecisionTreeRegressor(max_depth=4)
model2.fit(x_train,y_train)
model2.score(x_test,y_test)


# In[17]:


model3=KNeighborsRegressor(n_neighbors=5)
model3.fit(x_train,y_train)
model3.score(x_test,y_test)


# In[18]:


model4=SVR(kernel='sigmoid')
model4.fit(x_train,y_train)
model4.score(x_test,y_test)


# In[19]:


model5=SVR(kernel='rbf')
model5.fit(x_train,y_train)
model5.score(x_test,y_test)


# In[20]:


scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)


# In[21]:


scaled_model=RandomForestRegressor()
scaled_model.fit(x_train,y_train)
scaled_model.score(x_test,y_test)


# In[22]:


y_pred=scaled_model.predict(x_test)


# In[23]:


y_pred


# In[24]:


xgmodel=XGBRegressor()
xgmodel.fit(x_train,y_train)
xgmodel.score(x_test,y_test)


# In[26]:


pickle.dump(xgmodel,open('model.pkl','wb'))


# In[27]:


model=pickle.load(open('model.pkl','rb'))


# In[ ]:




