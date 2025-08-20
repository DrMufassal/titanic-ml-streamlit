import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load dataset and model
df = pd.read_csv("data/titanic.csv")
model = pickle.load(open("model.pkl", "rb"))

st.title("🚢 Titanic Survival Prediction")
st.write("Predict survival of passengers using ML model.")

menu = st.sidebar.selectbox("Navigation", ["EDA", "Visualizations", "Prediction"])

# EDA Section
if menu == "EDA":
    st.subheader("Exploratory Data Analysis")
    st.write("Shape of dataset:", df.shape)
    st.write(df.head())
    st.write("Missing values:")
    st.write(df.isnull().sum())

# Visualization Section
elif menu == "Visualizations":
    st.subheader("Visualizations")
    fig, ax = plt.subplots()
    sns.countplot(x="Survived", data=df, ax=ax)
    st.pyplot(fig)

    fig2 = px.histogram(df, x="Age", color="Survived", nbins=30)
    st.plotly_chart(fig2)

    fig3 = px.box(df, x="Pclass", y="Age", color="Survived")
    st.plotly_chart(fig3)

# Prediction Section
elif menu == "Prediction":
    st.subheader("Passenger Survival Prediction")

    # User inputs
    Pclass = st.selectbox("Class", [1, 2, 3])
    Age = st.slider("Age", 0, 80, 25)
    SibSp = st.number_input("Siblings/Spouses Aboard", 0, 8, 0)
    Parch = st.number_input("Parents/Children Aboard", 0, 6, 0)
    Fare = st.number_input("Fare", 0, 500, 50)
    Sex_male = st.selectbox("Sex", ["Female", "Male"])
    Embarked_Q = st.selectbox("Embarked Q?", [0, 1])
    Embarked_S = st.selectbox("Embarked S?", [0, 1])

    Sex_male = 1 if Sex_male == "Male" else 0

    features = pd.DataFrame([[Pclass, Age, SibSp, Parch, Fare, Sex_male, Embarked_Q, Embarked_S]],
                             columns=["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex_male", "Embarked_Q", "Embarked_S"])
    
    # Align columns with model training
    expected_cols = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex_male", "Embarked_Q", "Embarked_S"]
    features = features[expected_cols]

    prediction = model.predict(features)[0]
    st.write("Prediction:", "Survived 🟢" if prediction == 1 else "Not Survived 🔴")
