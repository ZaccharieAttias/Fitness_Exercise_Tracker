import itertools

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from LearningAlgorithms import ClassificationAlgorithms
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------
# Constants
# --------------------------------------------------------------
DATA_PATH = "../../data/interim/03_data_features.pkl"
OUTPUT_DATA_PATH = "../../data/processed/04_final_dataset.pkl"
OUTPUT_MODEL_PATH = "../../models/"

# --------------------------------------------------------------
# Plot settings
# --------------------------------------------------------------
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

# --------------------------------------------------------------
# Functions
# --------------------------------------------------------------


def split_data(df):
    """
    Split the data into training and test sets.

    Parameters:
    df (DataFrame): The dataset to be split.

    Returns:
    x_train (DataFrame): The training features.
    x_test (DataFrame): The test features.
    y_train (Series): The training labels.
    """
    df_train = df.drop(["participant", "category", "set"], axis=1)
    x, y = df_train.drop("label", axis=1), df_train["label"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y
    )
    return x_train, x_test, y_train, y_test


def visualize_label_distribution(df_train, y_train, y_test):
    """
    Visualize the distribution of the labels.

    Parameters:
    df_train (DataFrame): The training dataset.
    y_train (Series): The training labels.
    y_test (Series): The test labels.

    Returns:
    None
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    df_train["label"].value_counts().plot(
        kind="bar", ax=ax, color="lightblue", label="Total"
    )
    y_train.value_counts().plot(kind="bar", ax=ax, color="dodgerblue", label="Train")
    y_test.value_counts().plot(kind="bar", ax=ax, color="royalblue", label="Test")
    plt.legend()
    plt.show()


def define_feature_sets(df_train):
    """
    Define different sets of features.

    Parameters:
    df_train (DataFrame): The training dataset.

    Returns:
    feature_set1 (list): The first set of features.
    feature_set2 (list): The second set of features.
    feature_set3 (list): The third set of features.
    feature_set4 (list): The fourth set of features.
    """
    basic_features = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
    square_features = ["acc_r", "gyr_r"]
    pca_features = ["pca_1", "pca_2", "pca_3"]
    time_features = [f for f in df_train.columns if "_temp_" in f]
    freq_features = [f for f in df_train.columns if ("_freq" in f) or ("_pse" in f)]
    cluster_features = ["cluster"]

    feature_set1 = list(set(basic_features))
    feature_set2 = list(set(feature_set1 + square_features + pca_features))
    feature_set3 = list(set(feature_set2 + time_features))
    feature_set4 = list(set(feature_set3 + freq_features + cluster_features))

    return feature_set1, feature_set2, feature_set3, feature_set4


def forward_feature_selection(learner, x_train, y_train, max_features=10):
    """
    Perform forward feature selection.

    Parameters:
    learner (object): The classification algorithm object.
    x_train (DataFrame): The training features.
    y_train (Series): The training labels.
    max_features (int): The maximum number of features to select.

    Returns:
    selected_features (list): The selected features.
    ordered_features (list): The ordered features.
    """
    selected_features, ordered_features, ordered_scores = learner.forward_selection(
        max_features, x_train, y_train
    )
    return selected_features, ordered_scores


def plot_feature_selection_results(ordered_scores, max_features):
    """
    Plot the results of the feature selection.

    Parameters:
    ordered_scores (list): The ordered scores.
    max_features (int): The maximum number of features.

    Returns:
    None
    """
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, max_features + 1, 1), ordered_scores)
    plt.xlabel("Number of Features")
    plt.ylabel("Accuracy")
    plt.xticks(np.arange(1, max_features + 1, 1))
    plt.show()


def evaluate_models(
    learner, x_train, y_train, x_test, y_test, feature_sets, feature_names, iterations=1
):
    """
    Evaluate different models using the defined feature sets.

    Parameters:
    learner (object): The classification algorithm object.
    x_train (DataFrame): The training features.
    y_train (Series): The training labels.
    x_test (DataFrame): The test features.
    y_test (Series): The test labels.
    feature_sets (list): The list of feature sets.
    feature_names (list): The list of feature set names.
    iterations (int): The number of iterations to run.

    Returns:
    score_df (DataFrame): The model evaluation scores.
    """
    score_df = pd.DataFrame()
    for i, f in zip(range(len(feature_sets)), feature_names):
        print("Feature set:", i)
        selected_train_X = x_train[feature_sets[i]]
        selected_test_X = x_test[feature_sets[i]]

        # First run non deterministic classifiers to average their score.
        performance_test_nn = 0
        performance_test_rf = 0

        for it in range(0, iterations):
            print("\tTraining neural network,", it)
            (
                model,
                class_train_y,
                class_test_y,
                class_train_prob_y,
                class_test_prob_y,
            ) = learner.feedforward_neural_network(
                selected_train_X,
                y_train,
                selected_test_X,
                gridsearch=False,
            )
            performance_test_nn += accuracy_score(y_test, class_test_y)

            print("\tTraining random forest,", it)
            (
                model,
                class_train_y,
                class_test_y,
                class_train_prob_y,
                class_test_prob_y,
            ) = learner.random_forest(
                selected_train_X, y_train, selected_test_X, gridsearch=True
            )
            performance_test_rf += accuracy_score(y_test, class_test_y)

        performance_test_nn = performance_test_nn / iterations
        performance_test_rf = performance_test_rf / iterations

        # And we run our deterministic classifiers:
        print("\tTraining KNN")
        (
            model,
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.k_nearest_neighbor(
            selected_train_X, y_train, selected_test_X, gridsearch=True
        )
        performance_test_knn = accuracy_score(y_test, class_test_y)

        print("\tTraining decision tree")
        (
            model,
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.decision_tree(
            selected_train_X, y_train, selected_test_X, gridsearch=True
        )
        performance_test_dt = accuracy_score(y_test, class_test_y)

        print("\tTraining naive bayes")
        (
            model,
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.naive_bayes(selected_train_X, y_train, selected_test_X)

        performance_test_nb = accuracy_score(y_test, class_test_y)

        # Save results to dataframe
        models = ["NN", "RF", "KNN", "DT", "NB"]
        new_scores = pd.DataFrame(
            {
                "model": models,
                "feature_set": f,
                "accuracy": [
                    performance_test_nn,
                    performance_test_rf,
                    performance_test_knn,
                    performance_test_dt,
                    performance_test_nb,
                ],
            }
        )
        score_df = pd.concat([score_df, new_scores])

    print("Model evaluation completed.")
    return score_df


def plot_model_comparison(score_df):
    """
    Plot a comparison of the model performances.

    Parameters:
    socre_df (DataFrame): The model evaluation scores.

    Returns:
    None
    """
    score_df.sort_values(by="accuracy", ascending=False)
    plt.figure(figsize=(10, 10))
    sns.barplot(x="model", y="accuracy", hue="feature_set", data=score_df)
    plt.legend(loc="lower right")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.ylim(0.7, 1)
    plt.show()


def evaluate_best_model(
    score_df, learner, x_train, y_train, x_test, y_test, feature_set, gridsearch=True
):
    """
    Evaluate the best model using the selected feature set.

    Parameters:
    score_df (DataFrame): The model evaluation scores.
    learner (object): The classification algorithm object.
    x_train (DataFrame): The training features.
    y_train (Series): The training labels.
    x_test (DataFrame): The test features.
    y_test (Series): The test labels.
    feature_set (list): The selected feature set.
    gridsearch (bool): Whether to perform grid search.

    Returns:
    best_model (str): The best model.
    best_feature_set (str): The best feature set.
    model (object): The trained model.
    class_test_y (Series): The test labels.
    class_test_prob_y (DataFrame): The test probabilities.
    accuracy (float): The accuracy of the model.
    """

    if score_df.empty:
        print("No model evaluation data available.")
        return None
    else:
        best_model = score_df.sort_values(by="accuracy", ascending=False).iloc[0]
        print("Best model:", best_model["model"])
        print("Feature set:", best_model["feature_set"])
        if best_model["model"] == "NN":
            (
                model,
                class_train_y,
                class_test_y,
                class_train_prob_y,
                class_test_prob_y,
            ) = learner.feedforward_neural_network(
                x_train[feature_set],
                y_train,
                x_test[feature_set],
                gridsearch=gridsearch,
            )
        elif best_model["model"] == "RF":
            (
                model,
                class_train_y,
                class_test_y,
                class_train_prob_y,
                class_test_prob_y,
            ) = learner.random_forest(
                x_train[feature_set],
                y_train,
                x_test[feature_set],
                gridsearch=gridsearch,
            )
        elif best_model["model"] == "KNN":
            (
                model,
                class_train_y,
                class_test_y,
                class_train_prob_y,
                class_test_prob_y,
            ) = learner.k_nearest_neighbor(
                x_train[feature_set],
                y_train,
                x_test[feature_set],
                gridsearch=gridsearch,
            )
        elif best_model["model"] == "DT":
            (
                model,
                class_train_y,
                class_test_y,
                class_train_prob_y,
                class_test_prob_y,
            ) = learner.decision_tree(
                x_train[feature_set],
                y_train,
                x_test[feature_set],
                gridsearch=gridsearch,
            )
        elif best_model["model"] == "NB":
            (
                model,
                class_train_y,
                class_test_y,
                class_train_prob_y,
                class_test_prob_y,
            ) = learner.naive_bayes(x_train[feature_set], y_train, x_test[feature_set])
        else:
            print("Model not found.")
            return None

    accuracy = accuracy_score(y_test, class_test_y)
    return (
        best_model["model"],
        best_model["feature_set"],
        model,
        class_test_y,
        class_test_prob_y,
        accuracy,
    )


def plot_confusion_matrix(
    y_test, class_test_y, class_test_prob_y, model_name, accuracy
):
    """
    Plot the confusion matrix.

    Parameters:
    y_test (Series): The test labels.
    class_test_y (Series): The predicted labels.
    class_test_prob_y (DataFrame): The predicted probabilities.
    model_name (str): The name of the model.
    accuracy (float): The accuracy of the model.

    Returns:
    None
    """
    classes = class_test_prob_y.columns
    conf_matrix = confusion_matrix(y_test, class_test_y, labels=classes)
    plt.figure(figsize=(10, 10))
    plt.imshow(conf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(
        "Confusion matrix for " + model_name + " (Accuracy: " + str(accuracy) + ")"
    )
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = conf_matrix.max() / 2.0
    for i, j in itertools.product(
        range(conf_matrix.shape[0]), range(conf_matrix.shape[1])
    ):
        plt.text(
            j,
            i,
            format(conf_matrix[i, j]),
            horizontalalignment="center",
            color="white" if conf_matrix[i, j] > thresh else "black",
        )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.grid(False)
    plt.show()


def split_data_by_participant(df, df_train, participant):
    """
    Split the data into training and test sets based on participant.

    Parameters:
    df (DataFrame): The dataset to be split.
    df_train (DataFrame): The training dataset.
    participant (str): The participant to be excluded.

    Returns:
    x_train (DataFrame): The training features.
    x_test (DataFrame): The test features.
    y_train (Series): The training labels.
    y_test (Series): The test labels.
    """
    participant_df = df.drop(["category", "set"], axis=1)

    x_train = participant_df[participant_df["participant"] != participant].drop(
        "label", axis=1
    )
    y_train = participant_df[participant_df["participant"] != participant]["label"]

    x_test = participant_df[participant_df["participant"] == participant].drop(
        "label", axis=1
    )
    y_test = participant_df[participant_df["participant"] == participant]["label"]

    x_train = x_train.drop(["participant"], axis=1)
    x_test = x_test.drop(["participant"], axis=1)

    # Visualize the distribution of the labels
    fig, ax = plt.subplots(figsize=(10, 5))
    df_train["label"].value_counts().plot(
        kind="bar", ax=ax, color="lightblue", label="Total"
    )
    y_train.value_counts().plot(kind="bar", ax=ax, color="dodgerblue", label="Train")
    y_test.value_counts().plot(kind="bar", ax=ax, color="royalblue", label="Test")
    plt.legend()
    plt.show()

    return x_train, x_test, y_train, y_test


def export_model(model, dataset, feature_set):
    """
    Export the given model to the specified file path.

    Parameters:
    model (object): The machine learning model to be exported.
    dataset (object): The dataset used to train the model.
    feature_set (list): The list of features used to train the model.
    file_path (str): The path where the model will be saved.

    Returns:
    None
    """
    columns = feature_set
    target = "label"

    joblib.dump(
        value=[model, columns, target],
        filename=OUTPUT_MODEL_PATH + "MLPClassifier_model.joblib",
    )


def main():
    """Main function to load data, train models, and evaluate results."""
    df = pd.read_pickle(DATA_PATH)
    x_train, x_test, y_train, y_test = split_data(df)
    visualize_label_distribution(df, y_train, y_test)
    # split_data_by_participant(df, df, "A")
    feature_set1, feature_set2, feature_set3, feature_set4 = define_feature_sets(df)
    learner = ClassificationAlgorithms()
    selected_features, ordered_scores = forward_feature_selection(
        learner, x_train, y_train
    )
    plot_feature_selection_results(ordered_scores, max_features=10)
    feature_sets = [
        feature_set1,
        feature_set2,
        feature_set3,
        feature_set4,
        selected_features,
    ]
    feature_names = [
        "Feature Set 1",
        "Feature Set 2",
        "Feature Set 3",
        "Feature Set 4",
        "Selected Features",
    ]
    score_df = evaluate_models(
        learner, x_train, y_train, x_test, y_test, feature_sets, feature_names
    )
    plot_model_comparison(score_df)

    best_model = score_df.sort_values(by="accuracy", ascending=False).iloc[0]
    best_feature_set = best_model["feature_set"]
    feature_set_mapping = {
        "Feature Set 1": feature_set1,
        "Feature Set 2": feature_set2,
        "Feature Set 3": feature_set3,
        "Feature Set 4": feature_set4,
        "Selected Features": selected_features,
    }
    feature_set = feature_set_mapping.get(best_feature_set)

    best_model, best_feature_set, model, class_test_y, class_test_prob_y, accuracy = (
        evaluate_best_model(
            score_df, learner, x_train, y_train, x_test, y_test, feature_set
        )
    )
    plot_confusion_matrix(y_test, class_test_y, class_test_prob_y, best_model, accuracy)

    export_model(model, dataset=df.reset_index(), feature_set=feature_set)


if __name__ == "__main__":
    main()
