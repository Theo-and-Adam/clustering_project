import os

import pandas as pd

def read_and_combine_wine_data():
    """
    Read the red and white wine DataFrames from the provided URLs, add a 'wine_type' column to each,
    and combine them into one DataFrame.

    Returns:
    - df: Combined DataFrame containing both red and white wine data.
    """
    # Read red and white wine DataFrames
    red_url = 'https://query.data.world/s/km3v7y3hbhnq6q3qxmgunsty22chem?dws=00000'
    white_url = 'https://query.data.world/s/tzchvcxc66f2wiye4k3x3agz2vthyu?dws=00000'
    red_df = pd.read_csv(red_url)
    white_df = pd.read_csv(white_url)

    # Add a 'wine_type' column to each DataFrame
    red_df['wine_type'] = 'red'
    white_df['wine_type'] = 'white'

    # Combine the DataFrames into one
    df = pd.concat([red_df, white_df], ignore_index=True)

    return df
