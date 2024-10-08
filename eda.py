import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px 

import statsmodels.api as sm

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import metrics

from pathlib import Path

import streamlit as st

################ Preprocessing ####################

# Function to get the filtered data set
def get_data():

    # usage of pathlib
    PROJECT_DIR = Path(__file__).parent
    path = PROJECT_DIR / 'data/autoscout24.csv'
    data = pd.read_csv(path)

    return data


# Function to get the missing values
def get_missing_values(data):
    # calculating missing values
    missing_values = data.isnull().sum()
    missing_percentages = 100 * data.isnull().sum() / len(data)

    # generate new DataFrame
    missing_table = pd.DataFrame({
        "Abs. missing values": missing_values,
        "Percentage missing": missing_percentages.round(2)
    })

    return missing_table


# Function to get data without nan-values
def get_data_without_na(data):

    # drop NaN values from features with object-type: "model" and "gear" 
    data = data.dropna(subset=["model", "gear"])

    # replace NaN values from features with numeric-type with median value: "hp"
    data["hp"] = data["hp"].fillna(data["hp"].median())

    return data

################### Start Explorative Data Analysis ###################

# if chosen "Explorative Data Analysis"
def app():
    
    # Initialise preprocessed data without nan values
    data = get_data_without_na(get_data())

    st.info(":orange[AutoScout24 Trends and Insights:]\n\nThis visualisations provide a comprehensive overview of the data and offer a snapshot of the german car market.")

    ################ key metrics ##################

    # showing key metrics (adding expander later?)
    st.header("Key metrics")
    
    # slider filter for range of years
    year_min, year_max = int(data['year'].min()), int(data['year'].max())
    year_range = st.slider(
        "Select range of 'years of cars registration' to restrict data selection:",
        min_value=year_min,
        max_value=year_max,
        value=(year_min, year_max),
        step=1,
        key='year_range'
    )

    # Layout with five columns
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])

    # define filter-mask - range of registration years
    mask = data["year"].isin(range(year_range[0],year_range[1]+1))

    # Define what appears in the first column
    with col1:
        st.metric(":blue[:orange[No.] of car brands:]", value=data[mask]["make"].nunique())
        st.divider()
        st.metric(":blue[:orange[No.]. of car models:]", value=data[mask]["model"].nunique())

    with col2:
        st.metric(":blue[:orange[Ø] value of km driven:]", value=f"{data[mask]['mileage'].mean().__round__():,} km")
        st.divider()
        st.metric(":blue[:orange[Median] value of km driven:]", value=f"{data[mask]['mileage'].median().__round__():,} km")

    with col3:
        st.metric(":blue[:orange[Ø] value of horsepower:]", value=f"{data[mask]['hp'].mean().__round__():,} hp")
        st.divider()
        st.metric(":blue[:orange[Median] value of horsepower:]", value=f"{data[mask]['hp'].median().__round__():,} hp")

    with col4:
        st.metric(":blue[:orange[Ø] price:]", value=f"{data[mask]['price'].mean().__round__():,} €")
        st.divider()
        st.metric(":blue[:orange[Median] price:]", value=f"{data[mask]['price'].median().__round__():,} €")

    with col5:
        st.metric(":blue[:orange[Min.] price:]", value=f"{data[mask]['price'].max().__round__():,} €")
        st.divider()
        st.metric(":blue[:orange[Max.] price:]", value=f"{data[mask]['price'].min().__round__():,} €")

    st.divider()


    # import the data locally
    df = data 

    ############### Histogram and Average Value Visualisation ###############

    st.header("Visualisation of distributions and average values")
    st.info("Please do not miss the :orange[general filter] setting on the leftside / sidebar.")

    st.sidebar.subheader("General filter setting")

    # Layout with 2 columns
    #col1, col2 = st.columns([1, 1])


    hue = st.sidebar.selectbox("1.) Which :orange[qualitative feature] would you like to take a closer look at? Please select:",
                        ("Car brand", "Fuel type", "Gear type"), placeholder="Select qualitative feature")

    st.sidebar.info(":orange[Please note:] By default, all qualitative values are included. Use the filter below to restrict the selection.")

    if hue == "Car brand":
        color = "make"
        legend_title = "Car brand"

        with st.sidebar.popover("Select specific car brands"):
            selected_brand = st.multiselect("Select specific car brands", options= df.make.unique().tolist(), default=df.make.unique().tolist(), placeholder="car brand")
            df = df[df["make"].isin(selected_brand)]
            mask = df["make"].isin(selected_brand)

    if hue == "Fuel type":
        color = "fuel"
        legend_title = "Fuel Type"

        with st.sidebar.popover("Select specific fuel type"):
            selected_fuel = st.multiselect("Select specific fuel type", options= df.fuel.unique().tolist(), default=df.fuel.unique().tolist(), placeholder="fuel type")
            df = df[df["fuel"].isin(selected_fuel)]
            mask = df["fuel"].isin(selected_fuel)

    if hue == "Gear type":
        color = "gear"
        legend_title = "Gear Type"

        with st.sidebar.popover("Select specific gear type"):
            selected_gear = st.multiselect("Select specific gear type", options= df.gear.unique().tolist(), default=df.gear.unique().tolist(), placeholder="gear type")
            df = df[df["gear"].isin(selected_gear)]     
            mask = df["gear"].isin(selected_gear)

    # Layout with two columns
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Distribution of number of cars per...")
        
        dist = st.selectbox("2.a) For which :orange[quantitative feature] would you like to see the distribution? Please select:",
                                ("price", "year"))

        if dist == "year":
            x = 'year'
            xt = "Year"

        if dist == "price":
            x = 'price'
            xt = "Price in €"


        # visualizing histogram
        fig = px.histogram(df, x=x, hover_data=df.columns, marginal="box", color=color) 

        fig.update_layout(title=dict(text = f"Distribution of number of cars per {x}", font = dict(size = 25)),
                            title_x = 0.35,
                            barmode = "stack",
                            xaxis_title = xt,
                            yaxis_title = "Number of cars",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                bargap=0.2, # gap between bars of adjacent location coordinates
                legend=dict(title=dict(text=legend_title)))
        st.plotly_chart(fig)

        #st.divider()

    with col2:
            
        st.subheader("Average Values per 'year of car registration'")

        average_value = st.selectbox("2.b) Which :orange[average value] would you like to see per 'year of car registration'? Please select:", 
                                ("Average price", "Average mileage", "Average hp"))

        if average_value == "Average price":
            y = 'price'
            yt = "Average price in €"

        if average_value == "Average mileage":
            y = 'mileage'
            yt = "Average mileage in Km"

        if average_value == "Average hp":
            y = 'hp'
            yt = "Average hp"

        # initiliase local df
        filtered_df = df[mask]
        df2 = filtered_df.groupby(["year", "make", "fuel", "gear"]).agg({
            "mileage": "mean",
            "price": "mean",
            "hp": "mean"
        }).reset_index()

        df_yearly_avg = df2.groupby("year").agg({
        "mileage": "mean",
        "price": "mean",
        "hp": "mean"
        }).reset_index()

        # visualizing average value
        fig2 = px.bar(df_yearly_avg, x="year", y=y, hover_data=["mileage", "price", "hp"]) 

        fig2.update_layout(title=dict(text = f"Average '{y}' per 'year of car registration'", font = dict(size = 25)),
                            title_x = 0.35,
                            xaxis_title = "Year of car registration",
                            yaxis_title = yt,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='black'),
                bargap=0.2, # gap between bars of adjacent location coordinates
                legend=dict(title=dict(text=legend_title)))
        st.plotly_chart(fig2)
    
    st.divider()

    ############### Scatter Visualisations ###############

    st.header("Scatter Plot Visualisation")
    st.subheader("Do we have a correlation? Check it out!")

    # Layout with two columns
    col1, col2 = st.columns([1, 5])

    with col1:

        df3 = df[mask]

        df_quantitative = df3[["price", "mileage", "hp"]]
        df_qualitative = df3[['make', 'year', 'fuel', "gear"]]  

        xaxis = st.selectbox("1.) Please select a feature for the x-axis:",
                            df_quantitative.columns, index=None, placeholder="Select a feture for x-axis")
        
        yaxis = st.selectbox("2.) Please select a feature for the y-axis:",
                            df_quantitative.columns, index=None, placeholder="Select a feture for y-axis")
        
        zaxis = st.selectbox("3.) Do you wish a differentiaton by colour?",
                            df_qualitative.columns, index=None, placeholder="If so, select a feature")
        
        trendline = st.selectbox("3.) Please select, if you want to have a trendline:",
                            ("Show trendline", "No trendline"), index=None)

    with col2:

        # visualizing correlation
        fig3 = px.scatter(
            df3, 
            x=xaxis, 
            y=yaxis,
            opacity=0.4, 
            color=zaxis,
            hover_data=df3.columns,
            trendline="ols" if trendline == "Show trendline" else None) 

        fig3.update_layout(title=dict(text = f"Scatter plot of {xaxis} and {yaxis}", font = dict(size = 25)),
                            title_x = 0.35)

        st.plotly_chart(fig3)
