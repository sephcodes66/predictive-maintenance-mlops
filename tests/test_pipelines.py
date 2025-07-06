
from src.train import train_model
from src.tune import tune_model

def test_train_pipeline():
    """
    Tests the full training pipeline with the test configuration.
    """
    try:
        train_model("config/test_config.yaml")
    except Exception as e:
        assert False, f"Training pipeline failed with exception: {e}"

def test_tune_pipeline():
    """
    Tests the full tuning pipeline with the test configuration.
    """
    try:
        tune_model("config/test_config.yaml")
    except Exception as e:
        assert False, f"Tuning pipeline failed with exception: {e}"
