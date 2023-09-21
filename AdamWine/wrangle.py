import os
import pandas as pd
import numpy as np
import seaborn as sns


def read_wine_data():
    """
    Read the red and white wine DataFrames from the provided URLs.

    Returns:
    - red_df: DataFrame containing red wine data.
    - white_df: DataFrame containing white wine data.
    """
    red_url = 'https://query.data.world/s/km3v7y3hbhnq6q3qxmgunsty22chem?dws=00000'
    white_url = 'https://query.data.world/s/tzchvcxc66f2wiye4k3x3agz2vthyu?dws=00000'
    
    red_df = pd.read_csv(red_url)
    white_df = pd.read_csv(white_url)
    
    return red_df, white_df

red_wine_data, white_wine_data = df()
