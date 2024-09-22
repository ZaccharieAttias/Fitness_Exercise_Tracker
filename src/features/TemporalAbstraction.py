import numpy as np
import pandas as pd


class NumericalAbstraction:
    """
    Class to abstract a history of numerical values we can use as an attribute.
    """

    def aggregate_value(self, aggregation_function):
        """
        Aggregates a list of values using the specified aggregation function.

        Args:
            aggregation_function (str): The aggregation function to use ('mean', 'max', 'min', 'median', 'std').

        Returns:
            function: The corresponding numpy function for the aggregation.
        """
        if aggregation_function == "mean":
            return np.mean
        elif aggregation_function == "max":
            return np.max
        elif aggregation_function == "min":
            return np.min
        elif aggregation_function == "median":
            return np.median
        elif aggregation_function == "std":
            return np.std
        else:
            raise ValueError(f"Invalid aggregation function: {aggregation_function}")

    def abstract_numerical(self, data_table, cols, window_size, aggregation_function):
        """
        Abstract numerical columns specified given a window size and an aggregation function.

        Args:
            data_table (pd.DataFrame): The input data table.
            cols (list): List of columns to abstract.
            window_size (int): The number of time points from the past considered.
            aggregation_function (str): The aggregation function to use ('mean', 'max', 'min', 'median', 'std').

        Returns:
            pd.DataFrame: The data table with new columns for the temporal data.
        """
        for col in cols:
            new_col_name = f"{col}_temp_{aggregation_function}_ws_{window_size}"
            data_table[new_col_name] = (
                data_table[col]
                .rolling(window_size)
                .apply(self.aggregate_value(aggregation_function), raw=True)
            )

        return data_table


# Example usage:
if __name__ == "__main__":
    # Example data
    data = {
        "time": pd.date_range(start="1/1/2022", periods=10, freq="T"),
        "value": np.random.randn(10),
    }
    df = pd.DataFrame(data)
    df.set_index("time", inplace=True)

    # Create an instance of NumericalAbstraction
    num_abs = NumericalAbstraction()

    # Abstract numerical data
    result_df = num_abs.abstract_numerical(
        df, ["value"], window_size=3, aggregation_function="mean"
    )
    print(result_df)
