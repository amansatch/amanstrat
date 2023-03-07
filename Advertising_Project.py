import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Advertising Data Analysis")

@st.cache
def load_data():
    adv = pd.read_csv("Advertising.csv")
    return adv

adv = load_data()

st.write("The first 5 rows of the dataset:")
st.dataframe(adv.head())

st.write("The dataset contains", adv.shape[0], "rows and", adv.shape[1], "columns.")

st.write("Summary statistics of the dataset:")
st.dataframe(adv.describe())

st.write("Missing values in the dataset:")
st.dataframe(adv.isnull().sum())

# outlier analysis
st.write("Outlier analysis of the dataset:")
st.write("TV:")
st.pyplot(sns.boxplot(adv['TV']))
st.write("Radio:")
st.pyplot(sns.boxplot(adv['Radio']))
st.write("Newspaper:")
st.pyplot(sns.boxplot(adv['Newspaper']))
st.write("Sales:")
st.pyplot(sns.boxplot(adv['Sales']))

st.write("Relationship between advertising expenses and sales:")
sns.pairplot(adv,x_vars=['TV','Radio','Newspaper'],y_vars=['Sales'],kind='scatter')
st.pyplot(plt)

st.write("Correlation heatmap of the dataset:")
sns.heatmap(adv.corr(),annot=True)
st.pyplot(plt)

X = adv['TV'].values.reshape(-1,1)
y = adv['Sales'].values.reshape(-1,1)
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.8,test_size=0.2)

regressor = LinearRegression()
regressor.fit(X_train,y_train)

st.write("Linear Regression Model Results:")
st.write("Intercept:", regressor.intercept_)
st.write("Coefficient:", regressor.coef_)

plt.scatter(X_train,y_train)
plt.plot(X_train,0.056*X_train+6.859,'black')
st.pyplot(plt)

y_pred = regressor.predict(X_test)

plt.scatter(X_test,y_test)
plt.plot(X_test,0.056*X_test+6.859,'black')
st.pyplot(plt)

