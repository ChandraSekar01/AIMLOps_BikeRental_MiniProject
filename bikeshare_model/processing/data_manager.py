import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import re
import joblib
import pandas as pd
import typing as t
from sklearn.pipeline import Pipeline

from bikeshare_model import __version__ as _version
from bikeshare_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config


def load_raw_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    return dataframe


def get_year_and_month(df):
    # convert 'dteday' column to Datetime datatype
    df['dteday'] = pd.to_datetime(df['dteday'], format='%Y-%m-%d')
    # Add new features 'yr' and 'mnth
    df['yr'] = df['dteday'].dt.year.astype(str)
    df['mnth'] = df['dteday'].dt.month_name()

    return df


def numerical_categorical_columns(df, target_col, unused_colms):
    numerical_features = []
    categorical_features = []

    tgt_unused_cols = unused_colms.append(target_col)
    print(tgt_unused_cols)

    for col in df.columns:
        if col not in tgt_unused_cols:
            if df[col].dtypes == 'float64':
                numerical_features.append(col)
            else:
                categorical_features.append(col)

    print('Number of numerical variables: {}'.format(len(numerical_features)),":" , numerical_features)
    print('Number of categorical variables: {}'.format(len(categorical_features)),":" , categorical_features)

    return numerical_features, categorical_features


# def pre_pipeline_preparation(*, df: pd.DataFrame) -> pd.DataFrame:
#     df = get_year_and_month(df)
#     print(config.model_config_.target)
#     print(config.model_config_.unused_fields)
#     num_fea, cat_fea = numerical_categorical_columns(df, 
#                                                      config.model_config_.target, 
#                                                      config.model_config_.unused_fields)
    
#     return df, num_fea, cat_fea


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines/models.
    """
    do_not_delete = files_to_keep + ["__init__.py", ".gitignore"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()


def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """Save trained pipeline.
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """

    # Prepare versioned save file name
    save_file_name = f"{config.app_config_.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)
    print("Model/pipeline trained successfully!")


def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model