import os
import logging
from pathlib import Path

# Logs
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] line-%(lineno)d %(name)s - %(levelname)s - %(message)s")

project_name = "MLProject"

list_of_file = [
    "bikeshare_model/config.yaml",
    "bikeshare_model/pipeline.py",
    "bikeshare_model/predict.py",
    "bikeshare_model/train_pipeline.py",
    "bikeshare_model/VERSION",
    "bikeshare_model/__init__.py",
    "bikeshare_model/config/__init__.py",
    "bikeshare_model/config/core.py",
    "bikeshare_model/dataset/bike-rental-dataset.csv",
    "bikeshare_model/dataset/__init__.py",
    "bikeshare_model/processing/__init__.py",
    "bikeshare_model/processing/data_manager.py",
    "bikeshare_model/processing/features.py",
    "bikeshare_model/processing/validation.py",
    "bikeshare_model/trained_models/__init__.py",
    "requirements/requirements.txt",
    "tests/__ini__.py",
    "tests/conftest.py",
    "tests/test_features.py",
    "tests/test_predictions.py",
    "MANIFEST.in",
    "mypy.ini",
    "pyproject.toml",
    "setup.py"
]

for filepath in list_of_file:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    print(filedir, filename)
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info("Creating directory - {}, for {}".format(filedir, filename))

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")

    else:
        logging.info("Files/Folder already exists")