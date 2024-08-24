import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# Load the default dataset
df_default = pd.read_csv('nyc_taxi.csv', parse_dates=['timestamp'])

# Streamlit app
st.markdown("<h1 style='text-align: center;'>Anomaly Detection Application with NYC Taxi Dataset</h1>", unsafe_allow_html=True)

st.info("""
**Dataset Information**:
- The dataset used in this application is the NYC Taxi dataset.
- The last data point in the dataset is from **January 31, 2015**.
- You can input new data to test from **February 01, 2015**.
""")

# Input new data section in the sidebar
st.sidebar.subheader("Input Your Data")

# Initialize session state to hold the new data inputs
if 'new_data' not in st.session_state:
    st.session_state.new_data = []

# Set default values for date and time
default_date = datetime(2015, 2, 1).date()
default_time = datetime(2015, 2, 1, 0, 0).time()

# Input fields with default values in the sidebar
new_date = st.sidebar.date_input("Select Date", value=default_date)
new_time = st.sidebar.time_input("Select Time", value=default_time)
new_value = st.sidebar.number_input("Enter Value", min_value=0.0, step=1.0)

# Button to add new data point in the sidebar
if st.sidebar.button("Add Data"):
    # Convert date and time to timestamp
    new_timestamp = pd.to_datetime(f"{new_date} {new_time}")
    # Append new data to the session state list
    st.session_state.new_data.append({'timestamp': new_timestamp, 'value': new_value})

# Display the table of new data points on the main page
st.subheader("Data Points Added")
df_new_data = pd.DataFrame(st.session_state.new_data)
st.dataframe(df_new_data)

# Button to submit and process the data on the main page
if st.button("Submit Data"):
    # Combine default and new data
    df_combined = pd.concat([df_default, df_new_data]).reset_index(drop=True)

    # Resample data to hourly
    hourly_data = df_combined.set_index('timestamp').resample('H').mean().reset_index()

    # Handle missing values by forward filling
    hourly_data['value'].fillna(method='ffill', inplace=True)

    # Group by Hour and Weekday
    hourly_data['Hour'] = hourly_data['timestamp'].dt.hour
    hourly_data['Weekday'] = hourly_data['timestamp'].dt.weekday

    # Z-Score Method
    df_z_score = hourly_data.copy()
    df_z_score['z_score'] = (df_z_score['value'] - df_z_score['value'].mean()) / df_z_score['value'].std()
    threshold = 3
    df_z_score['anomaly_z_score'] = np.where(np.abs(df_z_score['z_score']) > threshold, 1, 0)
    anomalies_z_score = df_z_score[df_z_score['anomaly_z_score'] == 1]

    # Isolation Forest
    scaler = MinMaxScaler()
    df_if = hourly_data.copy()
    df_if_scaled = scaler.fit_transform(df_if[['value']])
    iso_forest = IsolationForest(contamination=0.01, random_state=42)
    df_if['anomaly_if'] = iso_forest.fit_predict(df_if_scaled)
    df_if['anomaly_if'] = df_if['anomaly_if'].map({1: 0, -1: 1})
    anomalies_if = df_if[df_if['anomaly_if'] == 1]

    # Gaussian Probability
    df_gaussian = hourly_data.copy()
    mean = df_gaussian['value'].mean()
    std = df_gaussian['value'].std()
    df_gaussian['probability'] = norm.pdf(df_gaussian['value'], mean, std)
    threshold = df_gaussian['probability'].quantile(0.01)
    df_gaussian['anomaly_gaussian'] = np.where(df_gaussian['probability'] < threshold, 1, 0)
    anomalies_gaussian = df_gaussian[df_gaussian['anomaly_gaussian'] == 1]

    # One-Class SVM
    df_ocsvm = hourly_data.copy()
    df_ocsvm_scaled = scaler.fit_transform(df_ocsvm[['value']])
    ocsvm = OneClassSVM(gamma='auto', nu=0.01)
    df_ocsvm['anomaly_ocsvm'] = ocsvm.fit_predict(df_ocsvm_scaled)
    df_ocsvm['anomaly_ocsvm'] = df_ocsvm['anomaly_ocsvm'].map({1: 0, -1: 1})
    anomalies_ocsvm = df_ocsvm[df_ocsvm['anomaly_ocsvm'] == 1]

    # Plotting
    fig = make_subplots(rows=2, cols=2, 
                        subplot_titles=('Z-Score Method', 'Isolation Forest', 
                                        'Gaussian Probability', 'One-Class SVM'))

    # Z-Score Method
    fig.add_trace(go.Scatter(x=hourly_data['timestamp'], y=hourly_data['value'], mode='lines', name='Value'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=anomalies_z_score['timestamp'], y=anomalies_z_score['value'], mode='markers', 
                             marker=dict(color='purple'), name='Anomalies'),
                  row=1, col=1)

    # Isolation Forest
    fig.add_trace(go.Scatter(x=hourly_data['timestamp'], y=hourly_data['value'], mode='lines', name='Value'),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=anomalies_if['timestamp'], y=anomalies_if['value'], mode='markers', 
                             marker=dict(color='purple'), name='Anomalies'),
                  row=1, col=2)

    # Gaussian Probability
    fig.add_trace(go.Scatter(x=hourly_data['timestamp'], y=hourly_data['value'], mode='lines', name='Value'),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=anomalies_gaussian['timestamp'], y=anomalies_gaussian['value'], mode='markers', 
                             marker=dict(color='purple'), name='Anomalies'),
                  row=2, col=1)

    # One-Class SVM
    fig.add_trace(go.Scatter(x=hourly_data['timestamp'], y=hourly_data['value'], mode='lines', name='Value'),
                  row=2, col=2)
    fig.add_trace(go.Scatter(x=anomalies_ocsvm['timestamp'], y=anomalies_ocsvm['value'], mode='markers', 
                             marker=dict(color='purple'), name='Anomalies'),
                  row=2, col=2)

    # Highlight new data points with arrows
    for _, row in df_new_data.iterrows():
        fig.add_annotation(
            x=row['timestamp'], 
            y=row['value'], 
            text="New Data", 
            showarrow=True, 
            arrowhead=2, 
            ax=-20, 
            ay=-30,
            arrowcolor="blue"
        )

    # Update layout
    fig.update_layout(
        title_text='Anomaly Detection with Default and New Data',
        showlegend=True,
        height=800,  
        width=1200,  
    )

    # Display plot in Streamlit
    st.plotly_chart(fig)
