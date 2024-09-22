import copy

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, lfilter
from sklearn.decomposition import PCA


class LowPassFilter:
    """
    Class to remove high frequency data (considered noise) from the data.
    This can only be applied when there are no missing values (i.e., NaN).
    """

    def low_pass_filter(
        self,
        data_table,
        col,
        sampling_frequency,
        cutoff_frequency,
        order=5,
        phase_shift=True,
    ):
        """
        Apply a low-pass filter to the specified column in the data table.

        Args:
            data_table (pd.DataFrame): The input data table.
            col (str): The column to apply the filter to.
            sampling_frequency (float): The sampling frequency of the data.
            cutoff_frequency (float): The cutoff frequency for the filter.
            order (int, optional): The order of the filter. Defaults to 5.
            phase_shift (bool, optional): Whether to apply zero-phase filtering. Defaults to True.

        Returns:
            pd.DataFrame: The data table with the filtered column added.
        """
        nyquist_freq = 0.5 * sampling_frequency
        normalized_cutoff = cutoff_frequency / nyquist_freq

        b, a = butter(order, normalized_cutoff, btype="low", analog=False)
        if phase_shift:
            data_table[col + "_lowpass"] = filtfilt(b, a, data_table[col])
        else:
            data_table[col + "_lowpass"] = lfilter(b, a, data_table[col])
        return data_table


class PrincipalComponentAnalysis:
    """
    Class for Principal Component Analysis (PCA).
    This can only be applied when there are no missing values (i.e., NaN).
    """

    def __init__(self):
        self.pca = []

    def normalize_dataset(self, data_table, columns):
        """
        Normalize the specified columns in the data table.

        Args:
            data_table (pd.DataFrame): The input data table.
            columns (list): List of columns to normalize.

        Returns:
            pd.DataFrame: The normalized data table.
        """
        dt_norm = copy.deepcopy(data_table)
        for col in columns:
            dt_norm[col] = (data_table[col] - data_table[col].mean()) / (
                data_table[col].max() - data_table[col].min()
            )
        return dt_norm

    def determine_pc_explained_variance(self, data_table, cols):
        """
        Perform PCA on the selected columns and return the explained variance.

        Args:
            data_table (pd.DataFrame): The input data table.
            cols (list): List of columns to apply PCA to.

        Returns:
            np.ndarray: The explained variance ratio of the principal components.
        """
        dt_norm = self.normalize_dataset(data_table, cols)
        self.pca = PCA(n_components=len(cols))
        self.pca.fit(dt_norm[cols])
        return self.pca.explained_variance_ratio_

    def apply_pca(self, data_table, cols, number_comp):
        """
        Apply PCA to the selected columns and add new PCA columns to the data table.

        Args:
            data_table (pd.DataFrame): The input data table.
            cols (list): List of columns to apply PCA to.
            number_comp (int): The number of principal components to retain.

        Returns:
            pd.DataFrame: The data table with new PCA columns added.
        """
        dt_norm = self.normalize_dataset(data_table, cols)
        self.pca = PCA(n_components=number_comp)
        self.pca.fit(dt_norm[cols])
        new_values = self.pca.transform(dt_norm[cols])

        for comp in range(number_comp):
            data_table[f"pca_{comp + 1}"] = new_values[:, comp]

        return data_table


# Example usage:
if __name__ == "__main__":
    # Example data
    data = {
        "time": pd.date_range(start="1/1/2022", periods=100, freq="T"),
        "value": np.random.randn(100),
    }
    df = pd.DataFrame(data)
    df.set_index("time", inplace=True)

    # Apply LowPassFilter
    lpf = LowPassFilter()
    df_filtered = lpf.low_pass_filter(
        df, "value", sampling_frequency=1, cutoff_frequency=0.1
    )
    print(df_filtered.head())

    # Apply PrincipalComponentAnalysis
    pca = PrincipalComponentAnalysis()
    explained_variance = pca.determine_pc_explained_variance(
        df_filtered, ["value_lowpass"]
    )
    print("Explained variance:", explained_variance)

    df_pca = pca.apply_pca(df_filtered, ["value_lowpass"], number_comp=1)
    print(df_pca.head())
