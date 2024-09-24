import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from sklearn.neighbors import LocalOutlierFactor

# --------------------------------------------------------------
# Constants
# --------------------------------------------------------------
DATA_PATH = "../../data/interim/01_data_processed.pkl"
OUTPUT_PATH = "../../data/interim/02_outliers_removed_chauvenet.pkl"
OUTLIER_COLUMNS = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]

# --------------------------------------------------------------
# Plot settings
# --------------------------------------------------------------
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100

# --------------------------------------------------------------
# Classes
# --------------------------------------------------------------


class DataCleaner:
    """
    Data cleaning class to remove outliers from the dataset.
    """

    def __init__(self, data_path=DATA_PATH, output_path=OUTPUT_PATH):
        self.data_path = data_path
        self.output_path = output_path
        self.df = pd.read_pickle(self.data_path)

    def plot_binary_outliers(self, dataset, col, outlier_col, reset_index):
        """Plot outliers in case of a binary outlier score.

        Args:
            dataset (pd.DataFrame): The dataset
            col (string): Column that you want to plot
            outlier_col (string): Outlier column marked with true/false
            reset_index (bool): whether to reset the index for plotting
        """
        dataset = dataset.dropna(axis=0, subset=[col, outlier_col])
        dataset[outlier_col] = dataset[outlier_col].astype("bool")

        if reset_index:
            dataset = dataset.reset_index()

        fig, ax = plt.subplots()
        plt.xlabel("samples")
        plt.ylabel("value")

        # Plot non-outliers in default color
        ax.plot(
            dataset.index[~dataset[outlier_col]],
            dataset[col][~dataset[outlier_col]],
            "+",
        )
        # Plot data points that are outliers in red
        ax.plot(
            dataset.index[dataset[outlier_col]],
            dataset[col][dataset[outlier_col]],
            "r+",
        )

        plt.legend(
            ["no outlier " + col, "outlier " + col],
            loc="upper center",
            ncol=2,
            fancybox=True,
            shadow=True,
        )
        plt.show()

    def mark_outliers_iqr(self, dataset, col):
        """Mark values as outliers using the IQR method.

        Args:
            dataset (pd.DataFrame): The dataset
            col (string): The column you want to apply outlier detection to

        Returns:
            pd.DataFrame: The original dataframe with an extra boolean column indicating whether the value is an outlier or not.
        """
        dataset = dataset.copy()
        Q1 = dataset[col].quantile(0.25)
        Q3 = dataset[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        dataset[col + "_outlier"] = (dataset[col] < lower_bound) | (
            dataset[col] > upper_bound
        )
        return dataset

    def mark_outliers_chauvenet(self, dataset, col, C=2):
        """Mark values as outliers using Chauvenet's criterion.

        Args:
            dataset (pd.DataFrame): The dataset
            col (string): The column you want to apply outlier detection to
            C (int, optional): Degree of certainty for the identification of outliers given the assumption of a normal distribution. Defaults to 2.

        Returns:
            pd.DataFrame: The original dataframe with an extra boolean column indicating whether the value is an outlier or not.
        """
        dataset = dataset.copy()

        # Compute the mean and standard deviation.
        mean = dataset[col].mean()
        std = dataset[col].std()
        N = len(dataset.index)
        criterion = 1.0 / (C * N)

        # Consider the deviation for the data points.
        deviation = abs(dataset[col] - mean) / std

        # Express the upper and lower bounds.
        low = -deviation / math.sqrt(C)
        high = deviation / math.sqrt(C)
        prob = []
        mask = []

        # Pass all rows in the dataset.
        for i in range(len(dataset.index)):
            # Determine the probability of observing the point
            prob.append(
                1.0 - 0.5 * (scipy.special.erf(high[i]) - scipy.special.erf(low[i]))
            )
            # And mark as an outlier when the probability is below our criterion.
            mask.append(prob[i] < criterion)
        dataset[col + "_outlier"] = mask
        return dataset

    def mark_outliers_lof(self, dataset, columns, n=20):
        """Mark values as outliers using Local Outlier Factor (LOF).

        Args:
            dataset (pd.DataFrame): The dataset
            columns (list): The columns you want to apply outlier detection to
            n (int, optional): Number of neighbors to use. Defaults to 20.

        Returns:
            pd.DataFrame: The original dataframe with an extra boolean column indicating whether the value is an outlier or not.
        """
        dataset = dataset.copy()

        lof = LocalOutlierFactor(n_neighbors=n)
        data = dataset[columns]
        outliers = lof.fit_predict(data)

        dataset["outlier_lof"] = outliers == -1
        return dataset, outliers, lof.negative_outlier_factor_

    def remove_outliers(self, df, columns, method="chauvenet"):
        """Remove outliers from the dataframe using the specified method.

        Args:
            df (pd.DataFrame): The input dataframe
            columns (list): List of columns to apply outlier detection to
            method (str, optional): The method to use for outlier detection. Defaults to "chauvenet".

        Returns:
            pd.DataFrame: The dataframe with outliers removed
        """
        outliers_removed_df = df.copy()
        for col in columns:
            for label in df["label"].unique():
                if method == "iqr":
                    dataset = self.mark_outliers_iqr(df[df["label"] == label], col)
                elif method == "chauvenet":
                    dataset = self.mark_outliers_chauvenet(
                        df[df["label"] == label], col
                    )
                elif method == "lof":
                    dataset, _, _ = self.mark_outliers_lof(
                        df[df["label"] == label], columns
                    )
                else:
                    raise ValueError("Invalid method specified")

                dataset.loc[dataset[col + "_outlier"], col] = np.nan
                outliers_removed_df.loc[
                    (outliers_removed_df["label"] == label), col
                ] = dataset[col]

                n_outliers = len(dataset) - len(dataset[col].dropna())
                print(f"{n_outliers} outliers removed for {col} in {label}")

        return outliers_removed_df

    def data_cleaning(self, method="chauvenet"):
        """Main function to load data, remove outliers, and export the cleaned dataframe."""
        df = self.df

        # Plot initial data
        df[["gyr_y", "label"]].boxplot(by="label", figsize=(20, 10))
        df[OUTLIER_COLUMNS[:3] + ["label"]].boxplot(
            by="label", figsize=(20, 10), layout=(1, 3)
        )
        df[OUTLIER_COLUMNS[3:] + ["label"]].boxplot(
            by="label", figsize=(20, 10), layout=(1, 3)
        )

        # Remove outliers using Chauvenet's criterion
        outliers_removed_df = self.remove_outliers(df, OUTLIER_COLUMNS, method=method)

        # Export cleaned dataframe
        outliers_removed_df.to_pickle(self.output_path)
        print(f"Cleaned data saved to {self.output_path}")


def main():
    """Main function to load data, remove outliers, and export the cleaned dataframe."""
    datacleaner = DataCleaner(DATA_PATH, OUTPUT_PATH)
    datacleaner.data_cleaning(method="chauvenet")

    # df = pd.read_pickle(DATA_PATH)

    # # Plot initial data
    # df[["gyr_y", "label"]].boxplot(by="label", figsize=(20, 10))
    # df[OUTLIER_COLUMNS[:3] + ["label"]].boxplot(
    #     by="label", figsize=(20, 10), layout=(1, 3)
    # )
    # df[OUTLIER_COLUMNS[3:] + ["label"]].boxplot(
    #     by="label", figsize=(20, 10), layout=(1, 3)
    # )

    # # Remove outliers using Chauvenet's criterion
    # outliers_removed_df = remove_outliers(df, OUTLIER_COLUMNS, method="chauvenet")

    # # Export cleaned dataframe
    # outliers_removed_df.to_pickle(OUTPUT_PATH)
    # print(f"Cleaned data saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
