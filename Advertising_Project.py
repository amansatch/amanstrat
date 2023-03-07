import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LinearRegression

# Data Visualisation
import matplotlib.pyplot as plt 
import seaborn as sns

advertising = datasets.load_advertising(as_frame=True).data

st.write("""
# Simple Advertising Prediction App
This app predicts the **Advertising** type!
""")

st.sidebar.header('User Input Parameters')

# Data Visualization
st.subheader('Data Visualization')
fig, axs = plt.subplots(3, figsize = (5,5))
plt1 = sns.histplot(advertising['TV'], kde=True, ax = axs[0])
axs[0].set_title('TV')
plt2 = sns.histplot(advertising['Newspaper'], kde=True, ax = axs[1])
axs[1].set_title('Newspaper')
plt3 = sns.histplot(advertising['Radio'], kde=True, ax = axs[2])
axs[2].set_title('Radio')
st.pyplot(fig)

# Prediction Model
st.subheader('Prediction Model')
X = advertising[['TV', 'Newspaper', 'Radio']]
y = advertising['Sales']
model = LinearRegression()
model.fit(X, y)

# User Input
st.sidebar.subheader('User Input Parameters')
tv = st.sidebar.slider('TV advertising budget', float(advertising['TV'].min()), float(advertising['TV'].max()), float(advertising['TV'].mean()))
newspaper = st.sidebar.slider('Newspaper advertising budget', float(advertising['Newspaper'].min()), float(advertising['Newspaper'].max()), float(advertising['Newspaper'].mean()))
radio = st.sidebar.slider('Radio advertising budget', float(advertising['Radio'].min()), float(advertising['Radio'].max()), float(advertising['Radio'].mean()))

# Model Prediction
predicted_sales = model.predict([[tv, newspaper, radio]])

# Display Prediction
st.subheader('Sales Prediction')
st.write(f"The estimated sales for the given advertising budget are {predicted_sales[0]:.2f} units.")
