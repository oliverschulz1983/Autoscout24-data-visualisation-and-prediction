import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px 

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import metrics

from xgboost import XGBRegressor

from pathlib import Path

import streamlit as st

import eda

def select_dataset():
    # Define the list in the dataset selection
    content = [":blue[Complete dataset]",
        ":blue[Only Top 5 Brands] ('Volkswagen', 'Opel', 'Ford', 'Skoda' and 'Renault')"]   

    # initialise dataset selection in sidebar
    selected_dataset = st.radio(":blue[:orange[Select dataset] for Prediction:]", content)

    # define functions for initialising preprocessed datasets 
    def complete_dataset():
        # Initialise preprocessed data without nan values
        dataml = eda.get_data_without_na(eda.get_data())

        # define function to generate new column "age"
        def get_ages(x):
            x_new = 2021 - x
            return x_new

        # add new column "age" in ml dataset 
        dataml["age"] = dataml["year"].apply(get_ages)

        return dataml

    def limited_dataset():
        # Initialise preprocessed data without nan values
        dataml = eda.get_data_without_na(eda.get_data())

        # define function to generate new column "age"
        def get_ages(x):
            x_new = 2021 - x
            return x_new

        # add new column "age" in ml dataset 
        dataml["age"] = dataml["year"].apply(get_ages)

        mask = dataml["make"].isin(["Volkswagen", "Opel", "Ford", "Skoda", "Renault"])

        return dataml[mask]


    if selected_dataset == ":blue[Complete dataset]":
        return complete_dataset()

    elif selected_dataset == ":blue[Only Top 5 Brands] ('Volkswagen', 'Opel', 'Ford', 'Skoda' and 'Renault')":
        return limited_dataset()
    
    else:
        st.error("Invalid dataset selection")
        return None



def prediction(dataml, selected_data):

    # Define the features(X) and the label(y)
    X = dataml[["make", "model", "fuel", "gear", "offerType", "mileage", "hp", "age"]].reset_index(drop=True)
    y = dataml[["price"]].reset_index(drop=True)

    # Create the training and test data sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=100)

    # Preprocessing for categorical and numerical data using sklearn ColumnTransformer model
    preprocessor = ColumnTransformer(transformers= [
        ("num", StandardScaler(), ["mileage", "hp", "age"]),
        ("cat", OneHotEncoder(handle_unknown="ignore",drop='first'),["make", "model", "fuel", "gear", "offerType"])
    ]
    )
    # create a pipeline that combines preprocessing and model training 
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(n_estimators=100, random_state=42, max_depth=10))
    ])

    # format y_train
    y_train = y_train.squeeze()

    # fit the model on the training data
    pipeline.fit(X_train, y_train)

    return pipeline.predict(selected_data) # Make predictions

def input_data():

    dataml = select_dataset()

    if dataml is None:
        st.error("Failed to load dataset.")
        return

    st.info("Please :orange[select the features] for price prediction with 'Extreme Gradient Boosting Regression Model':")
    
    # Layout with 2 columns
    col1, col2 = st.columns([1, 1])

    # Define col1
    with col1:

        selected_make = st.selectbox("Brand", options=dataml["make"].unique().tolist())
        filtered_dataml = dataml[dataml["make"] == selected_make] # Filter the DataFrame for the selected brand

        selected_model = st.selectbox("Model", options=filtered_dataml["model"].unique().tolist())

        selected_fuel = st.selectbox("Fuel type", options=dataml["fuel"].unique().tolist())

        selected_gear = st.selectbox("Gear type", options=dataml["gear"].unique().tolist())

        selected_offertype = st.selectbox("Offer type", options=dataml["offerType"].unique().tolist())

    # Define col2
    with col2:
        hp_min, hp_max = int(dataml['hp'].min()), int(dataml['hp'].max())
        selected_hp = st.slider("Horse Power", min_value=1, max_value=999, value=60, step=1)

        selected_mileage = st.slider("Mileage", min_value=1, max_value=499999, value=10000, step=1)

        selected_age = st.slider("Age", min_value=1, max_value=49, value=5, step=1)

        selected_data = pd.DataFrame({
            "make": selected_make,
            "model": selected_model,
            "fuel": selected_fuel,
            "gear": selected_gear,
            "offerType": selected_offertype,
            "hp": selected_hp,
            "mileage": selected_mileage,
            "age": selected_age
        }, index=[0])

    predict = st.button("Predict price", help="Click this button to start the Extreme Gradient Boosting Regression Model based Prediction.")
    
    try:
        if predict:
            with st.spinner("Prediction in progress..."):
                # Predicting the price of the new car entry
                predicted_price = prediction(dataml, selected_data)

                st.metric("#### :green[The Extreme Gradient Boosting Model predicted price is:] ", f" {predicted_price[0]:,.2f} EUR")
                st.snow()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Debug info:")
        st.write("Selected data:", selected_data)
        st.write("DataML shape:", dataml.shape)
        st.write("DataML columns:", dataml.columns)


######################## Function that controls the structure ##########

# if chosen "Prediction" radio 
def app():    

    st.header("How much money could you get for your car? Check it out!")

    input_data()



 