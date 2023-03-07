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
x = df['TV']
y = df['Sales']

w0 = 7
w1 = 7

for i in range(0,50):
    yh = w0 + w1*x

    dew0 = -2 * ((y-yh)).mean()
    dew1 = -2 * ((y-yh) * x).mean()

    lr = 0.00001
    w0 = w0 - lr*dew0
    w1 = w1 - lr*dew1

    error = ((y-yh)**2).mean()
    print(i,error)


