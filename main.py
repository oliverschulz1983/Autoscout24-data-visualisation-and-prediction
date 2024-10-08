# Start with:
# streamlit run main.py

import streamlit as st

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import introduction
import eda
import ml
import pred

st.set_page_config(layout="wide") # Configure the Streamlit layout

# Customised CSS
st.markdown("""
    <style>
    div[data-baseweb="select"] > div {
        background-color: #FF7300; /* Background colour */
    }
    /* Font colour of the options */
    div[data-baseweb="select"] span {
        color: white !important; /* Font colour of the options */
    }
    /* Colour of the arrows in the drop-down menu */
    div[data-baseweb="select"] svg {
        fill: white; 
    }
    </style>
    """, unsafe_allow_html=True)

# initialise and get the raw data
data = eda.get_data()

# Main heading of the web app
st.title("'AutoScout24 - Germany Cars Dataset' - Visualisation and Prediction")

# Displaying the sidebar
st.sidebar.image("autoscout24logo.png")
st.sidebar.header("Navigation")

# Setup navigation in sidebar
#navigation = st.sidebar.radio('Go to:', ['Introduction', 'Explorative Data Analysis', 'Inferential', 'Machine Learning', 'Prediction'])

# Define the list in the navigation bar
content = {"Introduction": introduction,
    "Explorative Data Analysis": eda,
    "Machine Learning Evaluation": ml,
    "Price Prediction": pred}                                   

# Setup navigation in sidebar
select = st.sidebar.radio("Go to:", list(content.keys()))

content[select].app()