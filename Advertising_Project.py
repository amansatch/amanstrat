import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv('Advertising.csv')

st.write("""
# Simple Advertising Prediction App
This app predicts the **Advertising** type!
""")

st.sidebar.header('User Input Parameters')
st.write(df)
