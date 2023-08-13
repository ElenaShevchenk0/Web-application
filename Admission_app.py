#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 18:52:16 2020

@author: macpro
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
#pip install streamlit 
import streamlit as st

# add title to the app

st.header(""" Predict your chance to be accepted into Math Grad School """)



# user input data

def get_user_input():
    
    st.sidebar.title("Put your information here:")
    
    gpa = st.sidebar.slider('GPA (ratio of your GPA to max GPA)', min_value=0.0, max_value=1.0, value=0.5)
    gre_math = st.sidebar.slider('GRE Subject percentile (if no put zero)', min_value=0, 
                                 max_value=100, value=0)
    gre_quant = st.sidebar.slider('GRE General Quantitative percentile', 
                                  min_value=0, max_value=100, value=50)
    gre_verb = st.sidebar.slider('GRE General Verbal percentile', min_value=0, 
                                  max_value=100, value=50)
    gre_awa = st.sidebar.slider('GRE General AWA percentile', min_value=0, 
                                  max_value=100, value=50)
    major = st.sidebar.slider('Undegrad major (Pure math=1 / Applied math=2 / Others=3)', min_value=1, 
                                  max_value=3, step=1)
    type_st = st.sidebar.checkbox('Check if Domestic(US)')
    research = st.sidebar.checkbox('Check if you have research experience?')
    sex = st.sidebar.checkbox('Check if Female')
    
    user_data = {'gpa': gpa,  'gre_math': gre_math,  'gre_quant': gre_quant, 
                 'gre_verb': gre_verb, 'gre_awa': gre_awa, 'type_st': type_st,
                 'research': research, 'sex': sex, 'major': major}
    features = pd.DataFrame(user_data, index=[0])
    
    return features

user_input = get_user_input()

#st.subheader('User input:')
#st.write(user_input)



# Admission_math data

data = pd.read_csv("Admission_math")
data = data.drop(["Unnamed: 0"], axis=1)
data['gre_math'] = data['gre_math'].astype('category')
data['type_st'] = data['type_st'].astype('category')
data['research'] = data['research'].astype('category')
data['sex'] = data['sex'].astype('category')
data['major'] = data['major'].astype('category')
data['year'] = data['year'].astype('category')

# Split data into train and test sets 

data_train = data.sample(frac=0.8, random_state=3)
data_test = data[~data.index.isin(data_train.index)]

# Model for dummy math_gre

form = 'accept_rate ~ gpa + gre_math + gre_quant + gre_verb + gre_awa + type_st + research + sex + major'
log_model1 = smf.glm(formula=form, data=data_train, family=sm.families.Binomial(), \
                     var_weights=data_train['applications'])
res1 = log_model1.fit()

# MSE

pred = res1.predict(data_test)
MSE1 = sm.tools.eval_measures.mse(data_test['accept_rate'],pred)

# Admission_reduced data

df = pd.read_csv("Admission_reduced")
df = df.drop(["Unnamed: 0"], axis=1)
df['type_st'] = df['type_st'].astype('category')
df['research'] = df['research'].astype('category')
df['sex'] = df['sex'].astype('category')
df['major'] = df['major'].astype('category')
df['year'] = df['year'].astype('category')

# split data into train and test

df_train = df.sample(frac=0.8, random_state=12)
df_test = df[~df.index.isin(df_train.index)]

# Model for continuous math_gre

form = 'accept_rate ~ gpa + gre_math + gre_quant + gre_verb + gre_awa + type_st + research + sex + major'
logistic_model = smf.glm(formula=form, data=df_train, family=sm.families.Binomial(), \
                     var_weights=df_train['applications'])
results = logistic_model.fit()

# MSE

pred_test = results.predict(df_test)
MSE2 = sm.tools.eval_measures.mse(df_test['accept_rate'], pred_test)

# prediction

if user_input._get_value(0, 'gre_math') > 0:
   prediction = results.predict(user_input) 
   accuracy = 1-MSE2
else:
    prediction = res1.predict(user_input)
    accuracy = 1-MSE1

st.subheader('Prediction:')
st.write(format((prediction[0] * 100), '.2f') + '%')
st.subheader('Accuracy of prediction:')
st.write(format(accuracy * 100, '.2f') + '%')

