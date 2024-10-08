import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px 

import statsmodels.api as sm

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import metrics

from pathlib import Path

import streamlit as st

import eda

################ visualisation functions ###############

# define function for visualisation of the correct vs. predicted labels
def visualise_evaluation_data(evaluation_data):
    fig = px.scatter(evaluation_data, x="correct", y="predicted", opacity=0.4) 

    fig.update_layout(title=dict(text = f"Correct vs. predicted prices", font = dict(size = 25)),
                        title_x = 0.35,
                        xaxis_title = 'Correct labels: y_test',
                        yaxis_title = 'Predicted labels: y_pred',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'))
    st.plotly_chart(fig)

# define function for visualisation of evaluation metrics
def visualise_metrics(metrics):
     col1, col2, col3 = st.columns([1, 1, 1])

     with col1:
        st.metric(":orange[MAE]", value=metrics["MAE"][0])

     with col2:
        st.metric(":orange[RMSE]", value=metrics["RMSE"][0])

     with col3:
        st.metric(":orange[R-squared in %]", value=metrics["R-squared in %"][0])


def show_results(model):
    # usage of pathlib
    PROJECT_DIR = Path(__file__).parent
    path1 = PROJECT_DIR / f'data/{model}_evaluation_data.csv'
    path2 = PROJECT_DIR / f'data/{model}_metrics.csv'
        
    evaluation_data = pd.read_csv(path1)  # Load the model evaluation results 
    metrics = pd.read_csv(path2)  # Load the metric results of the model

    visualise_metrics(metrics)
    visualise_evaluation_data(evaluation_data) 

################ ML Result Visualisations ###########

# define all functions to visualise each model results
def lin_reg_evaluation():            
        st.subheader("Linear Regression")

        with st.expander("More details:"):
            st.write(":orange[Regulation:] No.")
            st.write(":orange[Preprocessing:] Transformed 'year' (of registration) => 'age' (of car).")
            st.write(":orange[Standardisation:] Not necessary because of no regulation.")
            st.write(":orange[Considered features:] Only numeric => 'mileage', 'hp' and 'age'")
            st.write(":orange[Categorial Variables:] Not taken into account.")
            st.write(":orange[Target:] 'price'")

        model = "lin_reg"

        show_results(model)

def ridge_reg_evaluation():
        st.subheader("Linear Ridge Regression")

        with st.expander("More details:"):
            st.write(":orange[Regulation:] Yes => Alpha = 1.0")
            st.write(":orange[Preprocessing:] Transformed 'year' (of registration) => 'age' (of car).")
            st.write(":orange[Standardisation:] Applied.")
            st.write(":orange[Considered features:] 'mileage', 'hp', 'age', 'make', 'model', 'fuel', 'gear' and 'offerType'")
            st.write(":orange[Categorial Variables:] Transforemd to 'Dummy'-Variables.")
            st.write(":orange[Target:] 'price'")

        model = 'ridge_reg'

        show_results(model)

def lasso_reg_evaluation():
        st.subheader("Linear Lasso Regression")

        with st.expander("More details:"):
            st.write(":orange[Regulation:] Yes => Alpha determined by cross-validation => Alpha = 28.48")
            st.write(":orange[Preprocessing:] Transformed 'year' (of registration) => 'age' (of car).")
            st.write(":orange[Standardisation:] Applied.")
            st.write(":orange[Considered features:] 'mileage', 'hp', 'age', 'make', 'model', 'fuel', 'gear' and 'offerType'")
            st.write(":orange[Categorial Variables:] Transforemd to 'Dummy'-Variables.")
            st.write(":orange[Target:] 'price'")

        model = 'lasso_reg'

        show_results(model)

def elastic_reg_evaluation():
        st.subheader("Linear Elastic Net Regression")

        with st.expander("More details:"):
            st.write(":orange[Regulation:] Yes => Alpha and l1-ratio determined by cross-validation => Alpha = 0.00001, l1-ratio = 0.5")
            st.write(":orange[Preprocessing:] Transformed 'year' (of registration) => 'age' (of car).")
            st.write(":orange[Standardisation:] Applied.")
            st.write(":orange[Considered features:] 'mileage', 'hp', 'age', 'make', 'model', 'fuel', 'gear' and 'offerType'")
            st.write(":orange[Categorial Variables:] Transforemd to 'Dummy'-Variables.")
            st.write(":orange[Target:] 'price'")

        model = 'elastic_reg'

        show_results(model)

def tree_reg_evaluation():
        st.subheader("Decision Tree Regression")

        with st.expander("More details:"):
            st.write(":orange[Preprocessing:] Transformed 'year' (of registration) => 'age' (of car).")
            st.write(":orange[Standardisation:] Not applied.")
            st.write(":orange[Max depth:] Determined by manual testing => 10")
            st.write(":orange[Considered features:] 'mileage', 'hp', 'age', 'make', 'model', 'fuel', 'gear' and 'offerType'")
            st.write(":orange[Categorial Variables:] Transforemd to 'Dummy'-Variables.")
            st.write(":orange[Target:] 'price'")

        model = 'tree_reg'

        show_results(model)

def randomforest_reg_evaluation():
        st.subheader("Random Forest Regression")

        with st.expander("More details:"):
            st.write(":orange[Preprocessing:] Transformed 'year' (of registration) => 'age' (of car).")
            st.write(":orange[Standardisation:] Not applied.")
            st.write(":orange[Max depth:] Determined by manual testing => 10")
            st.write(":orange[No. of estimators:] Determined by manual testing => 100")
            st.write(":orange[Considered features:] 'mileage', 'hp', 'age', 'make', 'model', 'fuel', 'gear' and 'offerType'")
            st.write(":orange[Categorial Variables:] Transforemd to 'Dummy'-Variables.")
            st.write(":orange[Target:] 'price'")

        model = 'randomforest_reg'

        show_results(model)

def gb_reg_evaluation():
        st.subheader("Gradient Boosting Regression")

        with st.expander("More details:"):
            st.write(":orange[Preprocessing:] Transformed 'year' (of registration) => 'age' (of car).")
            st.write(":orange[Standardisation:] Not applied.")
            st.write(":orange[Max depth:] Determined by manual testing => 10")
            st.write(":orange[No. of estimators:] Determined by manual testing => 100")
            st.write(":orange[Considered features:] 'mileage', 'hp', 'age', 'make', 'model', 'fuel', 'gear' and 'offerType'")
            st.write(":orange[Categorial Variables:] Transforemd to 'Dummy'-Variables.")
            st.write(":orange[Target:] 'price'")

        model = 'gb_reg'

        show_results(model)

def xgb_reg_evaluation():
        st.subheader("Extreme Gradient Boosting Regression")

        with st.expander("More details:"):
            st.write(":orange[Preprocessing:] Transformed 'year' (of registration) => 'age' (of car).")
            st.write(":orange[Standardisation:] Applied for numeric variables.")
            st.write(":orange[Max depth:] Determined by manual testing => 10")
            st.write(":orange[No. of estimators:] Determined by manual testing => 100")
            st.write(":orange[Considered features:] 'mileage', 'hp', 'age', 'make', 'model', 'fuel', 'gear' and 'offerType'")
            st.write(":orange[Categorial Variables:] Transforemd to 'Dummy'-Variables via 'OneHotEncoder''.")
            st.write(":orange[Target:] 'price'")

        model = 'xgb_reg'

        show_results(model)

######################## Function that controls the structure ##########

# if chosen "Machine Learning" radio show model results
def app():    

    st.header("Evaluation of various Regression Models for Machine Learning")

    st.info("The following metrics and visualisations provide an overview of the explanatory power of various regression models for this dataset.")

    # Layout with 2 columns
    col1, col2 = st.columns([1, 1])

    # Define col1
    with col1:
        lin_reg_evaluation()
    
    # Define col2
    with col2:
        ridge_reg_evaluation()

    # Layout with 2 columns
    col1, col2 = st.columns([1, 1])

    # Define col1
    with col1:
        lasso_reg_evaluation()
    
    # Define col2
    with col2:
        elastic_reg_evaluation()

    # Layout with 2 columns
    col1, col2 = st.columns([1, 1])

    # Define col1
    with col1:
        tree_reg_evaluation()
    
    # Define col2
    with col2:
        randomforest_reg_evaluation()

    # Layout with 2 columns
    col1, col2 = st.columns([1, 1])

    # Define col1
    with col1:
        gb_reg_evaluation()
    
    # Define col2
    with col2:
        xgb_reg_evaluation()
