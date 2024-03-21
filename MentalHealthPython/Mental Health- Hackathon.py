#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import VotingRegressor
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from catboost import CatBoostClassifier
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import VotingRegressor
os.getcwd()


# In[2]:





# In[3]:


df = pd.read_csv('smmh.csv')
df.drop(['Timestamp','7. What social media platforms do you commonly use?',],axis=1,inplace=True)
y=target=df['18. How often do you feel depressed or down?']
df.drop(['18. How often do you feel depressed or down?'],axis=1,inplace=True)
df.columns = [col.split('. ')[1] if '. ' in col else col for col in df.columns]
df.drop(['What type of organizations are you affiliated with?'],axis=1,inplace=True)
a=LabelEncoder()
df['Gender']=a.fit_transform(df['Gender'])
df['Relationship Status']=a.fit_transform(df['Relationship Status'])
df['Occupation Status']=a.fit_transform(df['Occupation Status'])
df['Do you use social media?']=a.fit_transform(df['Do you use social media?'])
df['What is the average time you spend on social media every day?']=a.fit_transform(df['What is the average time you spend on social media every day?'])

rf_model = RandomForestClassifier()
rf_model.fit(df, y)


X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train,y_train)

catboost_classifier = CatBoostClassifier(iterations=100, learning_rate=0.1, verbose=False)
catboost_classifier.fit(X_train,y_train)



rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
catboost_classifier = CatBoostClassifier(iterations=100, learning_rate=0.1, verbose=False)

rf_classifier.fit(X_train, y_train)
catboost_classifier.fit(X_train, y_train)

ensemble_model1 = VotingClassifier(estimators=[('Random Forest', rf_classifier), ('CatBoost', catboost_classifier)], voting='hard')  

ensemble_model1.fit(X_train, y_train)



catboost_model = CatBoostRegressor()  
catboost_model.fit(X_train, y_train)  

xgboost_model = XGBRegressor()  
xgboost_model.fit(X_train, y_train) 

ensemble_model = VotingRegressor([('CatBoost', catboost_model), ('XGBoost', xgboost_model)])
ensemble_model.fit(X_train, y_train) 


user_inputs = [
    int(input("What is your age?: ")),
    int(input("Gender (1 for Male, 2 for Female, 3 for Other): ")),
    int(input("Relationship Status (1 for Single, 2 for In a Relationship, 3 for Married): ")),
    int(input("Occupation Status (1 for Employed, 2 for Unemployed, 3 for Student): ")),
    int(input("Do you use social media? (1 for Yes, 0 for No): ")),
    int(input("What is the average time you spend on social media every day?: ")),
    int(input("How often do you find yourself using Social media without a specific purpose? (1-5): ")),
    int(input("How often do you get distracted by Social media when you are busy doing something? (1-5): ")),
    int(input("Do you feel restless if you haven't used Social media in a while? (1 for Yes, 0 for No): ")),
    int(input("On a scale of 1 to 5, how easily distracted are you?: ")),
    int(input("On a scale of 1 to 5, how much are you bothered by worries?: ")),
    int(input("Do you find it difficult to concentrate on things? (1 for Yes, 0 for No): ")),
    int(input("On a scale of 1-5, how often do you compare yourself to other successful people through social media?: ")),
    int(input("Following the previous question, how do you feel about these comparisons, generally speaking? (1-5): ")),
    int(input("How often do you look to seek validation from features of social media? (1-5): ")),
    int(input("On a scale of 1 to 5, how frequently does your interest in daily activities fluctuate?: ")),
    int(input("On a scale of 1 to 5, how often do you face issues regarding sleep?: "))
]

if X_train.shape[0] > 1000 and X_train.shape[1] > 17:
    ensemble_predictions = ensemble_model.predict([user_inputs]) 
    print("The predicted class for the user using Ensemble Method is:", ensemble_predictions[0])
    
elif X_train.shape[0] <= 1000 and X_train.shape[1] <= 17:
    predicted_class = catboost_classifier.predict([user_inputs])  
    print("The predicted class for the user using CatBoost is:", predicted_class[0])


else:
    predicted_class=[3]
    print("The predicted class for the user using CatBoost is:", predicted_class[0])

