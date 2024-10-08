import streamlit as st
import pandas as pd 

import eda

################### Introduction ###################

# if chosen "introduction"
def app():
    
    # read the data
    data = eda.get_data()
    
    st.divider()
    url = "https://www.kaggle.com/datasets/ander289386/cars-germany/data"
    st.write(":orange[Data Source:] The data is sourced from Kaggle and provides scraped data from AutoScout24 with information about new and used cars. Please find [here](%s) the original data set 'Germany Cars Dataset'." % url)
    st.write(f":orange[Data Shape:] The Dataframe of the raw dataset has a shape of {data.shape[0]:,} Columns and {data.shape[1]} Rows. That means we have {data.shape[0]:,} cars offered and {data.shape[1]} features for every one of them.")
    st.write(":orange[App Description:] This app provides a comprehensive analysis of the above mentioned dataset from Autoscout24. Methods of explorative data analysis and predective machine learning were applied. The dataset was subjected to several pre-processing steps, which are essential for further implementation.")
    st.divider()

    st.subheader("Overview of the features within the dataset")

    # Layout with two columns
    quantitative, qualitative = st.columns([5, 5])

    # Define what appears in the first column
    with quantitative:
        st.info("Quantitative features")
        st.write(":orange['mileage':] Kilometres traveled by the vehicle.")
        st.write(":orange['price':] Sale price of the vehicle in EUR.")
        st.write(":orange['hp':] Horse power.")
        st.write(":orange['year':] The vehicle registration year.")

    # Define what appears in the second column 
    with qualitative:
        st.info("Qualitative features")
        st.write(":orange['make':] Brand of the car.")
        st.write(":orange['model':] Model of the car.")
        st.write(":orange['fuel':] Fuel type.")
        st.write(":orange['gear':] Manual or automatic.")
        st.write(":orange['offerType':] Type of offer (new, used, ...).")
    
    st.divider()

    st.subheader("Dealing with inconsistent and missing data")

    # Layout with two columns
    col1, col2 = st.columns([2, 6])

    # Define what appears in the first column
    with col1:
        # show missing values 
        st.info("Overview of missing values:")
        st.dataframe(eda.get_missing_values(data))

    # Define what appears in the second column 
    with col2:
        st.info("Features with missing values: This is how we deal with it.")
        st.write(":orange['model':] We cannot draw any conclusions as to which model the respective car is based on the given data, so we remove rows without a model designation.")
        st.write(":orange['gear':] We cannot draw any conclusions as to which gear the respective car is based on the given data, so we remove rows without a gear designation.")
        st.write(":orange['hp':] We cannot draw any conclusions as to how much hp the respective car is based on the given data, but as these are numerical values in this case, we can replace the missing values with the median value. By this way we lose less information and distort the data set as little as possible.")

    # showing the data without missing values
    st.info(f"After processing the dataset regarding the missing values the dataset now has a shape of {eda.get_data_without_na(data).shape[0]:,} Columns and {eda.get_data_without_na(data).shape[1]:,} Rows. Please find here an overview of the processed data:")

    st.dataframe(eda.get_data_without_na(data))
