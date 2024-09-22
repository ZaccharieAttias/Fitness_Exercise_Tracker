import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

# Plot settings
mpl.style.use("fivethirtyeight")
mpl.rcParams["figure.figsize"] = (20, 5)
mpl.rcParams["figure.dpi"] = 100
mpl.rcParams["lines.linewidth"] = 2


def plot_single_column(df, column, set_number):
    """
    Plot a single column for a specific set.

    Args:
        df (pd.DataFrame): The dataset.
        column (str): The column to plot.
        set_number (int): The set number to filter by.
    """
    set_df = df[df["set"] == set_number]
    plt.plot(set_df[column].reset_index(drop=True))
    plt.title(f"Set {set_number} - {column}")
    plt.xlabel("Samples")
    plt.ylabel(column)
    plt.show()


def plot_all_exercises(df, column):
    """
    Plot a specific column for all exercises.

    Args:
        df (pd.DataFrame): The dataset.
        column (str): The column to plot.
    """
    for label in df["label"].unique():
        subset = df[df["label"] == label]
        plt.plot(subset[column].reset_index(drop=True), label=label)
        plt.title(f"All Exercises - {column}")
        plt.xlabel("Samples")
        plt.ylabel(column)
        plt.legend()
        plt.show()


def compare_categories(df, label, column):
    """
    Compare a specific column for different categories within a label.

    Args:
        df (pd.DataFrame): The dataset.
        label (str): The exercise label to filter by.
        column (str): The column to plot.
    """
    category_df = df.query(f"label == '{label}'").reset_index()
    category_df.groupby(["category"])[column].plot()
    plt.title(f"{label} - {column} by Category")
    plt.xlabel("Samples")
    plt.ylabel(column)
    plt.legend()
    plt.show()


def compare_participants(df, label, column):
    """
    Compare a specific column for different participants within a label.

    Args:
        df (pd.DataFrame): The dataset.
        label (str): The exercise label to filter by.
        column (str): The column to plot.
    """
    participant_df = (
        df.query(f"label == '{label}'").sort_values("participant").reset_index()
    )
    participant_df.groupby(["participant"])[column].plot()
    plt.title(f"{label} - {column} by Participant")
    plt.xlabel("Samples")
    plt.ylabel(column)
    plt.legend()
    plt.show()


def plot_multiple_axes(df, label, participant):
    """
    Plot multiple axes for a specific label and participant.

    Args:
        df (pd.DataFrame): The dataset.
        label (str): The exercise label to filter by.
        participant (str): The participant to filter by.
    """
    all_axis_df = (
        df.query(f"label == '{label}'")
        .query(f"participant == '{participant}'")
        .reset_index()
    )
    fig, ax = plt.subplots()
    all_axis_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
    plt.title(f"{label} ({participant}) - Accelerometer Data")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Acceleration")
    plt.legend()
    plt.show()


def plot_combined(df, label, participant):
    """
    Plot combined accelerometer and gyroscope data for a specific label and participant.

    Args:
        df (pd.DataFrame): The dataset.
        label (str): The exercise label to filter by.
        participant (str): The participant to filter by.
    """
    combined_plot_df = (
        df.query(f"label == '{label}'")
        .query(f"participant == '{participant}'")
        .reset_index(drop=True)
    )
    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
    combined_plot_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax[0])
    combined_plot_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax[1])
    ax[0].set_title(f"{label} ({participant}) - Accelerometer Data")
    ax[1].set_title(f"{label} ({participant}) - Gyroscope Data")
    ax[0].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=3,
        fancybox=True,
        shadow=True,
    )
    ax[1].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=3,
        fancybox=True,
        shadow=True,
    )
    ax[1].set_xlabel("Samples")
    plt.show()


def save_combined_plots(df, output_dir):
    """
    Save combined accelerometer and gyroscope plots for all labels and participants.

    Args:
        df (pd.DataFrame): The dataset.
        output_dir (str): The directory to save the plots.
    """
    labels = df["label"].unique()
    participants = df["participant"].unique()

    for label in labels:
        for participant in participants:
            combined_plot_df = (
                df.query(f"label == '{label}'")
                .query(f"participant == '{participant}'")
                .reset_index()
            )
            if len(combined_plot_df) > 0:
                fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
                combined_plot_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax[0])
                combined_plot_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax[1])
                ax[0].legend(
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.15),
                    ncol=3,
                    fancybox=True,
                    shadow=True,
                )
                ax[1].legend(
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.15),
                    ncol=3,
                    fancybox=True,
                    shadow=True,
                )
                ax[1].set_xlabel("Samples")
                plt.savefig(f"{output_dir}/{label.title()} ({participant}).png")
                plt.close(fig)
