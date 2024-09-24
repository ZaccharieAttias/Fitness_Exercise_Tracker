import os
from glob import glob
from werkzeug.utils import secure_filename
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


import pandas as pd

# --------------------------------------------------------------
# Constants
# --------------------------------------------------------------

DATA_PATH = "../../data/raw/MetaMotion/"
OUTPUT_PATH = "../../data/interim/01_data_processed.pkl"
ACCELEROMETER_KEYWORD = "Accelerometer"
GYROSCOPE_KEYWORD = "Gyroscope"

# --------------------------------------------------------------
# Classes
# --------------------------------------------------------------


class DataProcessor:
    """
    Data processing class to read, process, merge, resample, and export the dataset.
    """

    def __init__(self, data_path=DATA_PATH, output_path=OUTPUT_PATH, data=None):
        self.data_path = data_path
        self.output_path = output_path
        # Save the list of files
        self.files = self.create_file_list() if data is None else data

    def create_file_list(self):
        """
        Create a list of all files in the data path.

        Returns:
            list: List of file paths.
        """
        files = glob(os.path.join(self.data_path, "*.csv"))
        return files

    def extract_features_from_filename(self, filename):
        """
        Extract participant, label, and category from the filename.

        Args:
            filename (str): The path to the file.

        Returns:
            tuple: participant, label, category
        """
        base_filename = os.path.basename(filename)
        participant = base_filename.split("-")[0]
        label = base_filename.split("-")[1]
        category = base_filename.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")
        return participant, label, category

    def read_data_from_files(self, files):
        """
        Read and process data from a list of files.

        Args:
            files (list): List of file paths.

        Returns:
            tuple: DataFrames for accelerometer and gyroscope data.
        """
        acc_df = pd.DataFrame()
        gyr_df = pd.DataFrame()
        acc_set = 1
        gyr_set = 1

        for file in files:
            participant, label, category = self.extract_features_from_filename(file)
            df = pd.read_csv(file)
            df["participant"] = participant
            df["label"] = label
            df["category"] = category

            if ACCELEROMETER_KEYWORD in file:
                df["set"] = acc_set
                acc_set += 1
                acc_df = pd.concat([acc_df, df])
            elif GYROSCOPE_KEYWORD in file:
                df["set"] = gyr_set
                gyr_set += 1
                gyr_df = pd.concat([gyr_df, df])

        acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
        gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

        acc_df.drop(columns=["epoch (ms)", "time (01:00)", "elapsed (s)"], inplace=True)
        gyr_df.drop(columns=["epoch (ms)", "time (01:00)", "elapsed (s)"], inplace=True)

        return acc_df, gyr_df

    def merge_datasets(self, acc_df, gyr_df):
        """
        Merge accelerometer and gyroscope datasets.

        Args:
            acc_df (DataFrame): Accelerometer data.
            gyr_df (DataFrame): Gyroscope data.

        Returns:
            DataFrame: Merged dataset.
        """
        data_merged = pd.concat([acc_df.iloc[:, :3], gyr_df], axis=1)

        # Rename columns for clarity
        data_merged.columns = [
            "acc_x",
            "acc_y",
            "acc_z",
            "gyr_x",
            "gyr_y",
            "gyr_z",
            "participant",
            "label",
            "category",
            "set",
        ]
        return data_merged

    def resample_data(self, data_merged):
        """
        Resample the merged dataset to a specified frequency.
        Our actual sampling frequencies are:
            Accelerometer:    12.500HZ
            Gyroscope:        25.000Hz

        Args:
            data_merged (DataFrame): Merged dataset.

        Returns:
            DataFrame: Resampled dataset.
        """
        sampling = {
            "acc_x": "mean",
            "acc_y": "mean",
            "acc_z": "mean",
            "gyr_x": "mean",
            "gyr_y": "mean",
            "gyr_z": "mean",
            "participant": "last",
            "label": "last",
            "category": "last",
            "set": "last",
        }

        days = [group for _, group in data_merged.groupby(pd.Grouper(freq="D"))]
        data_resampled = pd.concat(
            [df.resample(rule="200ms").apply(sampling).dropna() for df in days]
        )
        data_resampled["set"] = data_resampled["set"].astype("int64")
        return data_resampled

    def data_processing(self):
        """
        Main function to read, process, merge, resample, and export the dataset.
        """
        acc_df, gyr_df = self.read_data_from_files(self.files)
        data_merged = self.merge_datasets(acc_df, gyr_df)
        data_resampled = self.resample_data(data_merged)
        print("Data processing complete. Output saved to:", self.output_path)
        data_resampled.to_pickle(self.output_path)
        print("Data processing complete. Output saved to:", self.output_path)


def main():
    """
    Main function to read, process, merge, resample, and export the dataset.
    """
    processor = DataProcessor()
    processor.data_processing()
    # files = glob(os.path.join(DATA_PATH, "*.csv"))
    # acc_df, gyr_df = read_data_from_files(files)
    # data_merged = merge_datasets(acc_df, gyr_df)
    # data_resampled = resample_data(data_merged)
    # data_resampled.to_pickle(OUTPUT_PATH)
    # print("Data processing complete. Output saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
