import math
import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# define a function that will allow us to center around a particular radius
def filter_within_radius(df, center_lat, center_lon, radius_km):
    """
    Filters the input DataFrame for crimes within a specified radius (in km)
    from the given center latitude and longitude coordinates.

    Args:
    df (pandas.DataFrame): The DataFrame containing the crime data.
    center_lat (float): The latitude of the center point.
    center_lon (float): The longitude of the center point.
    radius_km (float): The radius within which to filter the crimes, in kilometers.

    Returns:
    pandas.DataFrame: A DataFrame filtered for crimes within the specified radius.
    """


    # Convert latitude and longitude differences to radians and calculate distances
    df['lat_diff'] = (df['Latitude'] - center_lat) * 111  # 111 km per degree of latitude
    df['lon_diff'] = ((df['Longitude'] - center_lon) * (math.pi / 180) * 111
                      * math.cos(center_lat * (math.pi / 180)))  # Adjust for latitude in longitude

    # Calculate the straight-line distance using Pythagorean theorem
    df['distance_km'] = (df['lat_diff']**2 + df['lon_diff']**2)**0.5

    # Filter for crimes within the specified radius
    df_filtered = df[df['distance_km'] <= radius_km]

    return df_filtered

# Define haversine function for radius calculation
def haversine(lat1, lon1, lat2, lon2):
    # Radius of the Earth in km
    R = 6371.0
    # Convert degrees to radians
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    # Differences in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    # Haversine formula
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance

# Function to apply text fixes using the dictionary
def apply_text_fixes(text, fix_map):
    for key, value in fix_map.items():
        text = text.replace(key, value)
    return text


import plotly.
def train_vs_test_plotly(y_train,y_test,y_pred):
    trace0 = go.Scatter(
        x=y_train.index,
        y=y_train,
        mode="lines+markers",
        name='Training',
        marker=dict(color='grey',size=10),
        line=dict(width=2))
    trace1 = go.Scatter(
        x=y_test.index,
        y=y_test,
        mode='lines+markers',
        name='Actual',
        marker=dict(color='blue',size=10),
        line=dict(width=2)
    )

    trace2 = go.Scatter(
        x=y_test.index,
        y=y_pred,
        mode='lines+markers',
        name='Predicted',
        marker=dict(color='red'),
        line=dict(width=1)
    )

    # Create the figure
    fig = go.Figure(data=[trace0,trace1, trace2])

    # Update layout
    fig.update_layout(
        title='Actual vs Predicted Numbers of Crime Throughout 2023',
        xaxis_title='Date',
        yaxis_title='Number of Crimes',
        legend_title='Legend',
        template="plotly"
    )
    fig.show()