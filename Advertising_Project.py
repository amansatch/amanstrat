import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

# Data Visualisation
import matplotlib.pyplot as plt 
import seaborn as sns

advertising = pd.read_csv('Advertising.csv')

st.write("""
# Simple Advertising Prediction App
This app predicts the **Advertising** type!
""")

st.sidebar.header('User Input Parameters')

# Outlier Analysis
fig, axs = plt.subplots(3, figsize = (5,5))
plt1 = sns.boxplot(advertising['TV'], ax = axs[0])
plt2 = sns.boxplot(advertising['Newspaper'], ax = axs[1])
plt3 = sns.boxplot(advertising['Radio'], ax = axs[2])
st.pyplot(plt1)

