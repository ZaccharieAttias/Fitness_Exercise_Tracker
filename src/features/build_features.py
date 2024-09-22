import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans
from TemporalAbstraction import NumericalAbstraction

# --------------------------------------------------------------
# Constants
# --------------------------------------------------------------
DATA_PATH = "../../data/interim/02_outliers_removed_chauvenet.pkl"
OUTPUT_PATH = "../../data/interim/03_data_features.pkl"
PREDICTOR_COLUMNS = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]

# --------------------------------------------------------------
# Functions
# --------------------------------------------------------------


def load_data(path):
    """Load data from a pickle file.

    Args:
        path (str): The path to the pickle file.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    return pd.read_pickle(path)


def impute_missing_values(df, columns):
    """Impute missing values using interpolation.

    Args:
        df (pd.DataFrame): The input data frame.
        columns (list): List of columns to impute missing values for.

    Returns:
        pd.DataFrame: The data frame with imputed values.
    """
    for col in columns:
        df[col] = df[col].interpolate()
    return df


def calculate_set_duration(df):
    """Calculate the duration of each set and add it as a new column.

    Args:
        df (pd.DataFrame): The input data frame.

    Returns:
        pd.DataFrame: The data frame with the duration of each set added as a new column.
    """
    for set_id in df["set"].unique():
        start = df[df["set"] == set_id].index[0]
        end = df[df["set"] == set_id].index[-1]
        duration = (end - start).seconds
        df.loc[df["set"] == set_id, "duration"] = duration
    return df


def apply_lowpass_filter(df, columns, fs, cutoff, order=5):
    """Apply a Butterworth lowpass filter to the specified columns.

    Args:
        df (pd.DataFrame): The input data frame.
        columns (list): List of columns to apply the filter to.
        fs (float): The sampling frequency.
        cutoff (float): The cutoff frequency.
        order (int, optional): The order of the filter. Defaults to 5.

    Returns:
        pd.DataFrame: The data frame with the filtered columns.
    """

    # Searching the best cutoff frequency

    # lpf = LowPassFilter()
    # df_lowpass = lpf.low_pass_filter(df_lowpass, "acc_y", fs, cutoff, order=5)

    # subset = df_lowpass[df_lowpass["set"] == 45]

    # fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
    # ax[0].plot(subset["acc_y"].reset_index(drop=True), label="raw data")
    # ax[1].plot(subset["acc_y_lowpass"].reset_index(drop=True), label="butterworth filter")
    # ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)
    # ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)

    lpf = LowPassFilter()
    for col in columns:
        df = lpf.low_pass_filter(df, col, fs, cutoff, order)
        df[col] = df[col + "_lowpass"]
        del df[col + "_lowpass"]
    return df


def apply_pca(df, columns, n_components):
    """Apply Principal Component Analysis (PCA) to the specified columns.

    Args:
        df (pd.DataFrame): The input data frame.
        columns (list): List of columns to apply PCA to.
        n_components (int): The number of principal components to retain.

    Returns:
        pd.DataFrame: The data frame with the PCA components added as new columns.
    """

    # Searching the best number of components

    # pca = PrincipalComponentAnalysis()
    # pc_values = pca.determine_pc_explained_variance(df, columns)

    # plt.figure(figsize=(10, 10))
    # plt.plot(range(1, len(columns) + 1), pc_values)
    # plt.xlabel("PCA numbers")
    # plt.ylabel("Explained Variance")
    # plt.show()

    pca = PrincipalComponentAnalysis()
    df = pca.apply_pca(df, columns, n_components)
    return df


def calculate_sum_of_squares(df):
    """Calculate the sum of squares for accelerometer and gyroscope data.

    Args:
        df (pd.DataFrame): The input data frame.

    Returns:
        pd.DataFrame: The data frame with the sum of squares added as new columns.
    """
    df["acc_r"] = np.sqrt(df["acc_x"] ** 2 + df["acc_y"] ** 2 + df["acc_z"] ** 2)
    df["gyr_r"] = np.sqrt(df["gyr_x"] ** 2 + df["gyr_y"] ** 2 + df["gyr_z"] ** 2)

    # Visualize results
    # subset = df[df["set"] == 45]
    # subset[["acc_r", "gyr_r"]].plot(subplots=True, figsize=(20, 10))
    return df


def apply_temporal_abstraction(df, columns, window_size):
    """Apply temporal abstraction to the specified columns.

    Args:
        df (pd.DataFrame): The input data frame.
        columns (list): List of columns to apply temporal abstraction to.
        window_size (int): The window size for the abstraction.

    Returns:
        pd.DataFrame: The data frame with the temporal abstraction added as new columns.
    """
    num_abs = NumericalAbstraction()
    for col in columns:
        df = num_abs.abstract_numerical(df, [col], window_size, "mean")
        df = num_abs.abstract_numerical(df, [col], window_size, "std")
    return df


def apply_frequency_abstraction(df, columns, window_size, fs):
    """Apply frequency abstraction to the specified columns.

    Args:
        df (pd.DataFrame): The input data frame.
        columns (list): List of columns to apply frequency abstraction to.
        window_size (int): The window size for the abstraction.
        fs (float): The sampling frequency.

    Returns:
        pd.DataFrame: The data frame with the frequency abstraction added as new columns.
    """
    df = df.reset_index()
    freq_abs = FourierTransformation()
    df_freq_list = []
    for set in df["set"].unique():
        print("Processing set: ", set)
        subset = df[df["set"] == set].reset_index(drop=True).copy()
        subset = freq_abs.abstract_frequency(subset, columns, window_size, fs)
        df_freq_list.append(subset)

    df = pd.concat(df_freq_list).set_index("epoch (ms)", drop=True)

    return df


def perform_clustering(df, columns, n_clusters):
    """Perform KMeans clustering on the specified columns.

    Args:
        df (pd.DataFrame): The input data frame.
        columns (list): List of columns to use for clustering.
        n_clusters (int): The number of clusters.

    Returns:
        pd.DataFrame: The data frame with the cluster labels added as a new column.
    """
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=0)
    df["cluster"] = kmeans.fit_predict(df[columns])
    return df


def visualize_clusters(df, columns):
    """Visualize the clusters in a 3D scatter plot.

    Args:
        df (pd.DataFrame): The input data frame.
        columns (list): List of columns to use for visualization.
    """

    # Searching the best number of clusters
    # k_values = range(2, 10)
    # inertias = []

    # for k in k_values:
    #     subset = df[columns]
    #     kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
    #     cluster_labels = kmeans.fit_predict(subset)
    #     inertias.append(kmeans.inertia_)

    # # Visualize results
    # plt.figure(figsize=(10, 10))
    # plt.plot(k_values, inertias)
    # plt.xlabel("k")
    # plt.ylabel("Sum of Squared distance")
    # plt.show()

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(projection="3d")
    for cluster in df["cluster"].unique():
        subset = df[df["cluster"] == cluster]
        ax.scatter(
            subset[columns[0]], subset[columns[1]], subset[columns[2]], label=cluster
        )
    ax.set_xlabel(columns[0])
    ax.set_ylabel(columns[1])
    ax.set_zlabel(columns[2])
    plt.legend()
    plt.show()


def visualize_labels(df, columns):
    """Visualize the labels in a 3D scatter plot.

    Args:
        df (pd.DataFrame): The input data frame.
        columns (list): List of columns to use for visualization.
    """
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(projection="3d")
    for label in df["label"].unique():
        subset = df[df["label"] == label]
        ax.scatter(
            subset[columns[0]], subset[columns[1]], subset[columns[2]], label=label
        )
    ax.set_xlabel(columns[0])
    ax.set_ylabel(columns[1])
    ax.set_zlabel(columns[2])
    plt.legend()
    plt.show()


def main():
    """Main function to build features from the dataset.

    This function orchestrates the entire feature engineering process, including:
    - Loading the data
    - Imputing missing values
    - Calculating set duration
    - Applying a lowpass filter
    - Applying PCA
    - Calculating the sum of squares
    - Applying temporal abstraction
    - Applying frequency abstraction
    - Performing clustering
    - Visualizing clusters and labels
    - Exporting the final dataset
    """
    # Load data
    df = load_data(DATA_PATH)

    # Impute missing values
    df = impute_missing_values(df, PREDICTOR_COLUMNS)

    # Calculate set duration
    # df = calculate_set_duration(df)

    # Apply lowpass filter
    fs = 1000 / 200
    cutoff = 1.3
    df = apply_lowpass_filter(df, PREDICTOR_COLUMNS, fs, cutoff)

    # Apply PCA
    df = apply_pca(df, PREDICTOR_COLUMNS, n_components=3)

    # Calculate sum of squares
    df = calculate_sum_of_squares(df)

    # Apply temporal abstraction
    window_size = int(1000 / 200)
    df = apply_temporal_abstraction(
        df, PREDICTOR_COLUMNS + ["acc_r", "gyr_r"], window_size
    )

    # Apply frequency abstraction
    window_size = int(2800 / 200)
    df = apply_frequency_abstraction(
        df, PREDICTOR_COLUMNS + ["acc_r", "gyr_r"], window_size, fs
    )

    # Dealing with overlapping windows
    df = df.dropna()

    # Perform clustering
    df = perform_clustering(df, ["acc_x", "acc_y", "acc_z"], n_clusters=5)

    # Visualize clusters
    visualize_clusters(df, ["acc_x", "acc_y", "acc_z"])

    # Visualize labels
    visualize_labels(df, ["acc_x", "acc_y", "acc_z"])

    # Export dataset
    df.to_pickle(OUTPUT_PATH)
    print(f"Data with features saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
