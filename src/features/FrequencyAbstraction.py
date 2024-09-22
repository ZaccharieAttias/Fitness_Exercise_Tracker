import numpy as np
import pandas as pd


class FourierTransformation:
    """
    Class to perform a Fourier transformation on the data to find frequencies that occur
    often and filter noise.
    """

    def find_fft_transformation(self, data, sampling_rate):
        """
        Find the amplitudes of the different frequencies using a fast Fourier transformation.

        Args:
            data (pd.Series): The input data series.
            sampling_rate (int): The number of samples per second (i.e., Frequency in Hertz of the dataset).

        Returns:
            tuple: Real and imaginary parts of the Fourier transformation.
        """
        transformation = np.fft.rfft(data, len(data))
        return transformation.real, transformation.imag

    def abstract_frequency(self, data_table, cols, window_size, sampling_rate):
        """
        Abstract frequency features from the data table.

        Args:
            data_table (pd.DataFrame): The input data table.
            cols (list): List of columns to abstract.
            window_size (int): The number of time points from the past considered.
            sampling_rate (int): The number of samples per second.

        Returns:
            pd.DataFrame: The data table with new columns for the frequency data.
        """
        # Create new columns for the frequency data.
        freqs = np.round((np.fft.rfftfreq(int(window_size)) * sampling_rate), 3)

        for col in cols:
            data_table[col + "_max_freq"] = np.nan
            data_table[col + "_freq_weighted"] = np.nan
            data_table[col + "_pse"] = np.nan
            for freq in freqs:
                data_table[
                    col + "_freq_" + str(freq) + "_Hz_ws_" + str(window_size)
                ] = np.nan

        # Pass over the dataset (we cannot compute it when we do not have enough history)
        # and compute the values.
        for i in range(window_size, len(data_table.index)):
            for col in cols:
                real_ampl, imag_ampl = self.find_fft_transformation(
                    data_table[col].iloc[
                        i - window_size : min(i + 1, len(data_table.index))
                    ],
                    sampling_rate,
                )
                # We only look at the real part in this implementation.
                for j in range(0, len(freqs)):
                    data_table.loc[
                        i, col + "_freq_" + str(freqs[j]) + "_Hz_ws_" + str(window_size)
                    ] = real_ampl[j]

                # Select the dominant frequency. We only consider the positive frequencies for now.
                data_table.loc[i, col + "_max_freq"] = freqs[
                    np.argmax(real_ampl[0 : len(real_ampl)])
                ]
                data_table.loc[i, col + "_freq_weighted"] = float(
                    np.sum(freqs * real_ampl)
                ) / np.sum(real_ampl)

                PSD = np.divide(np.square(real_ampl), float(len(real_ampl)))
                PSD_pdf = np.divide(PSD, np.sum(PSD))
                data_table.loc[i, col + "_pse"] = -np.sum(np.log(PSD_pdf) * PSD_pdf)

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

    # Create an instance of FourierTransformation
    fourier_transform = FourierTransformation()

    # Abstract frequency data
    result_df = fourier_transform.abstract_frequency(
        df, ["value"], window_size=10, sampling_rate=1
    )
    print(result_df)
