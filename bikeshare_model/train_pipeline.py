import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

from bikeshare_model.config.core import config
# from bikeshare_model.processing
from bikeshare_model.pipeline import bikeshare_pipe
from bikeshare_model.processing.data_manager import load_raw_dataset, save_pipeline, get_year_and_month

def run_training() -> None:
    """
    Training the model
    """
    try:
        # read training data
        data = load_raw_dataset(file_name=config.app_config_.training_data_file)
        # prepipeline fit
        df = get_year_and_month(data)
        print(df.head())
        print(df.info())
        print(df.describe())
        # df, _, _ = pre_pipeline_preparation(df = data)
        # df = data.copy()

        # train and test split
        print(config.model_config_.features)
        print(config.model_config_.target)
        X_train, X_test, y_train, y_test = train_test_split(
            df[config.model_config_.features],  # predictors
            df[[config.model_config_.target]],
            test_size=config.model_config_.test_size,
            random_state=config.model_config_.random_state,
        )
        print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

        # Pipeline fitting
        bikeshare_pipe.fit(X_train,y_train.values.ravel())
        y_pred = bikeshare_pipe.predict(X_test)
        # print("Accuracy(in %):", accuracy_score(y_test, y_pred)*100)
        print("R2 score:", r2_score(y_test, y_pred))
        print("Mean squared error:", mean_squared_error(y_test, y_pred))

        save_pipeline(pipeline_to_persist= bikeshare_pipe)
    except Exception as E:
        print(E)


if __name__ == "__main__":
    run_training()