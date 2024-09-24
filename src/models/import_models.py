import itertools

import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --------------------------------------------------------------
# Constants
# --------------------------------------------------------------
DATA_PATH = "../../data/processed/04_final_dataset.pkl"
MODEL_PATH = "../../models/MLPClassifier_model.joblib"
OUTOUT_CV_PATH = "../../reports/figures/cv_results.png"

# --------------------------------------------------------------
# Functions
# --------------------------------------------------------------


class LoadModel:
    """
    Load the model from the specified file on a specified dataset and evaluate the model.
    """

    def __init__(self, data_path=DATA_PATH, model_path=MODEL_PATH):
        self.data = self.load_data(data_path)
        self.model_path = model_path
        self.model, self.ref_columns, self.target = self.load_model(model_path)

    def load_model(self, model_path):
        """
        Load the model from the specified file.

        Parameters:
        model_path (str): The path to the model file.

        Returns:
        model: The loaded model.
        ref_columns: The reference columns.
        target: The target variable.
        """
        try:
            model, ref_columns, target = joblib.load(model_path)
            print(f"Model loaded successfully from {model_path}")
            return model, ref_columns, target
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def load_data(self, data_path):
        """
        Load the dataset from the specified file.

        Parameters:
        data_path (str): The path to the data file.

        Returns:
        data: The loaded dataset.
        """
        try:
            data = pd.read_pickle(data_path)
            print(f"Data loaded successfully from {data_path}")
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def evaluate_model(self, model, data, ref_columns, target):
        """
        Evaluate the model using the specified data.

        Parameters:
        model: The model to evaluate.
        data: The data to use for evaluation.
        ref_columns: The reference columns.
        target: The target variable.

        Returns:
        predictions: The predictions from the model.
        """
        X = data[ref_columns]
        y = data["label"]

        try:
            predictions = model.predict(X)
            accuracy = accuracy_score(y, predictions)
            report = classification_report(y, predictions)
            print(f"Accuracy: {accuracy}")
            print("Classification Report:")
            print(f"\n{report}")

            #self.plot_confusion_matrix(
                # y_test=y,
                # class_test_y=predictions,
                # class_test_prob_y=pd.DataFrame(
                #     model.predict_proba(X), columns=model.classes_
                # ),
                # model_name="MLPClassifier",
                # accuracy=accuracy,
            #)
        except Exception as e:
            print(f"Error during model evaluation: {e}")

        return predictions

    def plot_confusion_matrix(
        self, y_test, class_test_y, class_test_prob_y, model_name, accuracy
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
        plt.savefig("../../reports/figures/cv_results.png")

        plt.show()


def main():
    """ "Main function to load the model, load the data, and evaluate the model."""
    lm = LoadModel(data_path=DATA_PATH)
    lm.evaluate_model(
        model=lm.model, data=lm.data, ref_columns=lm.ref_columns, target=lm.target
    )

    # model, ref_columns, target = load_model(model_path=MODEL_PATH)
    # data = load_data(data_path=DATA_PATH)
    # evaluate_model(model=model, data=data, ref_columns=ref_columns, target=target)


if __name__ == "__main__":
    main()
