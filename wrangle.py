import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from IPython.display import clear_output
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer


def train_val_test(df):
    """
    Split the DataFrame into training, validation, and test sets.

    Returns:
    pd.DataFrame: Training, validation, and test DataFrames.
    """
    seed = 42
    train, val_test = train_test_split(df, train_size=0.7, random_state=seed)
    val, test = train_test_split(val_test, train_size=0.5, random_state=seed)
    
    # Return the three datasets
    return train, val, test


def get_dummies(train, val, test):
    """
    Convert specified columns into dummies and rename them.

    Parameters:
    train (pd.DataFrame): Training dataset.
    val (pd.DataFrame): Validation dataset.
    test (pd.DataFrame): Test dataset.

    Returns:
    train, val, test (pd.DataFrames): Modified datasets with dummies and renamed columns.
    """
    # Now you can proceed with one-hot encoding
    columns_to_convert = ['cluster']

    train=train.drop(columns=['density'])
    val=val.drop(columns=['density'])
    test=test.drop(columns=['density'])
    #'wine_type',
    # Perform conversion
    train = pd.get_dummies(train, columns=columns_to_convert)
    val = pd.get_dummies(val, columns=columns_to_convert)
    test = pd.get_dummies(test, columns=columns_to_convert)

    return train, val, test


def xy_split(df):
    """
    Split a DataFrame into features (X) and the target variable (y) by dropping the 'tax_value' column.

    Parameters:
    df (pd.DataFrame): DataFrame to be split.

    Returns:
    X (pd.DataFrame): Features.
    y (pd.Series): Target variable.
    """
    return df.drop(columns=['quality']), df.quality



def remove_outliers(df, k=1.5):
    '''
    remove_outliers will calculate the upper and lower fence
    based on the inner quartile range of each numerical column
    and remove rows with values outside of these fences.
    '''
    num_df = df.select_dtypes('number')
    
    for col in num_df.columns:
        q3 = num_df[col].quantile(0.75)
        q1 = num_df[col].quantile(0.25)
        iqr = q3 - q1
        upper_bound = q3 + (k * iqr)
        lower_bound = q1 - (k * iqr)
        
        # Remove rows with values outside of the fences
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    return df



def scaled_data(train, val, test, scaler_type='standard'):
    """
    Scale numerical features in train, val, and test datasets using various scaling techniques.

    Parameters:
    train (pd.DataFrame): Training dataset.
    val (pd.DataFrame): Validation dataset.
    test (pd.DataFrame): Test dataset.
    scaler_type (str): Type of scaler to use ('standard', 'minmax', 'robust', 'quantile').

    Returns:
    train, val, test (pd.DataFrames): Modified datasets with scaled numerical features.
    """
    # Initialize the selected scaler
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    elif scaler_type == 'quantile':
        scaler = QuantileTransformer(output_distribution='normal', random_state=42)
    else:
        raise ValueError("Invalid scaler_type. Choose from 'standard', 'minmax', 'robust', 'quantile'.")

    train = remove_outliers(train)
    val = remove_outliers(val)
    test = remove_outliers(test)

    # Exclude 'quality' from features to scale
    features_to_scale = train.columns.difference(['quality', 'cluster'])
    
    # Fit the scaler on the training data and transform all sets
    train[features_to_scale] = scaler.fit_transform(train[features_to_scale])
    val[features_to_scale] = scaler.transform(val[features_to_scale])
    test[features_to_scale] = scaler.transform(test[features_to_scale])

    return train, val, test


def model_df():
    data = {
        'Model': [
            'Linear Regression',
            'Random Forest Regressor',
            'Tweedie Regressor with Polynomial Features',
            'RandomForestRegressor with Polynomial Features',
            'XGBRegressor with Polynomial Features'
        ],
        'Train RMSE': [0.65, 0.21, 0.63, 0.21, 0.11],
        'Validate RMSE': [0.65, 0.54, 0.66, 0.55, 0.56]
    }
    
    df = pd.DataFrame(data)
    return df


def report_outliers(df, k=1.5) -> None:
    '''
    report_outliers will print a subset of each continuous
    series in a dataframe (based on numeric quality and n>20)
    and will print out results of this analysis with the fences
    in places
    '''
    num_df = df.select_dtypes('number')
    total_outliers = 0  # Initialize a variable to keep track of the total outliers
    for col in num_df:
        if len(num_df[col].value_counts()) > 20:
            lower_bound, upper_bound = get_fences(df, col, k=k)
            print(f'Outliers for Col {col}:')
            print('lower: ', lower_bound, 'upper: ', upper_bound)
            outliers = df[col][
                (df[col] > upper_bound) | (df[col] < lower_bound)]
            print(outliers)
            print('----------')
            total_outliers += len(outliers)  # Increment the total outlier count
    print(f"Total Outliers: {total_outliers}")  # Print the total outliers count
def get_fences(df, col, k=1.5) -> (float, float):
    '''
    get fences will calculate the upper and lower fence
    based on the inner quartile range of a single Series
    return: lower_bound and upper_bound, two floats
    '''
    q3 = df[col].quantile(0.75)
    q1 = df[col].quantile(0.25)
    iqr = q3 - q1
    upper_bound = q3 + (k * iqr)
    lower_bound = q1 - (k * iqr)
    return lower_bound, upper_bound


def cluster_plot(df):

    def random_centroids(data, k=3):
        centroids = []
        for i in range(k):
            centroid = data.apply(lambda x: float(x.sample()))
            centroids.append(centroid)
        return pd.concat(centroids, axis =1)
          #this wil define the cluster number
        #get labels
        #we can put this in a function
    def get_labels(df, centroids):
        distances = centroids.apply(lambda x: np.sqrt(((df -x)**2).sum(axis=1)))
        return distances.idxmin(axis=1)
        labels = get_labels(df, centroids)
        labels.value_counts()
        return labels
    
    #new centroids
    def new_centroids(data, label, k):
        return data.groupby(labels).apply(lambda x: np.exp(np.log(x).mean())).T
    
    #plot cluster
    def plot_clusters(data, labels, centroids, iteration):
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(data)
        centroids_2d = pca.transform(centroids.T)
        clear_output(wait=True)
        plt.title(f'Iteration {iteration}')
        # Plot data points with labels
        plt.scatter(x=data_2d[:, 0], y=data_2d[:, 1], c=labels, cmap='viridis')
        # Plot centroids as white squares
        plt.scatter(x=centroids_2d[:, 0], y=centroids_2d[:, 1], c='white', marker='s', s=100)
        plt.show()
    
    #write Kmeanalgorithm
    # write the body of Kmean algorithm
    max_iterations=100  #number of times the algorithm will iterate unless the cluster stop first
    k=5
    centroids=random_centroids(df, k)
    old_centroids=pd.DataFrame()
    iteration=1
    while iteration < max_iterations and not centroids.equals(old_centroids):
        old_centroids = centroids
        labels = get_labels(df, centroids)
        centroids = new_centroids(df, labels, k)
        plot_clusters(df, labels, centroids, iteration)
        iteration += 1