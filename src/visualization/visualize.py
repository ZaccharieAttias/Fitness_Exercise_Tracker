import pandas as pd
import visualization as viz

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/01_data_processed.pkl")

# --------------------------------------------------------------
# Plot single columns
# --------------------------------------------------------------

viz.plot_single_column(df, "acc_y", set_number=1)

# --------------------------------------------------------------
# Plot all exercises
# --------------------------------------------------------------

viz.plot_all_exercises(df, "acc_y")

# --------------------------------------------------------------
# Compare medium vs. heavy sets
# --------------------------------------------------------------

viz.compare_categories(df, label="squat", column="acc_y")

# --------------------------------------------------------------
# Compare participants
# --------------------------------------------------------------

viz.compare_participants(df, label="bench", column="acc_y")

# --------------------------------------------------------------
# Plot multiple axes
# --------------------------------------------------------------

viz.plot_multiple_axes(df, label="squat", participant="A")

# --------------------------------------------------------------
# Combine plots in one figure
# --------------------------------------------------------------

viz.plot_combined(df, label="row", participant="A")

# --------------------------------------------------------------
# Save combined plots for all labels and participants
# --------------------------------------------------------------

viz.save_combined_plots(df, output_dir="../../reports/figures")
