from glob import glob
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.make_dataset import DataProcessor
from src.features.remove_outliers import DataCleaner
from src.features.DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from src.features.FrequencyAbstraction import FourierTransformation
from src.features.TemporalAbstraction import NumericalAbstraction
from src.features.build_features import FeatureEngineering
from src.models.import_models import LoadModel


# --------------------------------------------------------------
# Constants
# --------------------------------------------------------------

DATA_PATH = "../../uploads/csv_files/"
OUTPUT_PATH = "../../uploads/processing_data/"

# --------------------------------------------------------------
# Functions
# --------------------------------------------------------------


def preprocess_data(files=None):
    """Preprocess the data for prediction."""
    data = DataProcessor(
        data_path=DATA_PATH,
        output_path=os.path.join(
            os.path.join(os.path.abspath(os.path.dirname(__file__)), "../.."),
            "uploads/processing_data/processed_data.pkl",
        ),
        data=files if files is not None else None,
    )
    data.data_processing()


def process_clean_data():
    """Process the cleaned data for prediction."""
    cleaned_data = DataCleaner(
        data_path=os.path.join(
            os.path.join(os.path.abspath(os.path.dirname(__file__)), "../.."),
            "uploads/processing_data/processed_data.pkl",
        ),
        output_path=os.path.join(
            os.path.join(os.path.abspath(os.path.dirname(__file__)), "../.."),
            "uploads/processing_data/cleaned_data.pkl",
        ),
    )
    cleaned_data.data_cleaning()


def adding_features():
    """Adding features to the cleaned data."""
    data_with_features = FeatureEngineering(
        data_path=os.path.join(
            os.path.join(os.path.abspath(os.path.dirname(__file__)), "../.."),
            "uploads/processing_data/cleaned_data.pkl",
        ),
        output_path=os.path.join(
            os.path.join(os.path.abspath(os.path.dirname(__file__)), "../.."),
            "uploads/processing_data/features_data.pkl",
        ),
    )
    data_with_features.build_new_features()


def test_model():
    """Test the model."""
    test_model = LoadModel(
        data_path=os.path.join(
            os.path.join(os.path.abspath(os.path.dirname(__file__)), "../.."),
            "uploads/processing_data/features_data.pkl",
        ),
        model_path=os.path.join(
            os.path.join(os.path.abspath(os.path.dirname(__file__)), "../.."),
            "models/MLPClassifier_model.joblib",
        ),
    )
    predictions = test_model.evaluate_model(
        test_model.model, test_model.data, test_model.ref_columns, test_model.target
    )
    return predictions


def main():
    """
    Main function to read, process, merge, resample, and export the dataset.
    """

    # Preprocess the data for prediction.
    preprocess_data()

    # Clean the data
    process_clean_data()

    # Add features to the cleaned data
    adding_features()

    # Test the model
    test_model()
