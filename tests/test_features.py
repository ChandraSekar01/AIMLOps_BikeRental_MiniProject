
"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np

from bikeshare_model.config.core import config

from bikeshare_model.processing.features import Mapper, WeekdayImputer, WeathersitImputer, WeekdayOneHotEncoder, OneHotEncoder, OutlierHandler


def test_weekday_imputer(sample_input_data):
    # Given
    imputer = WeekdayImputer()

    # When
    transformed_df = imputer.fit(sample_input_data[0]).transform(sample_input_data[0])

    # Then
    assert transformed_df['weekday'].iloc[0] == 'Sun' 

def test_weathersit_imputer(sample_input_data):
    # Given
    imputer = WeathersitImputer()

    # When
    transformed_df = imputer.fit(sample_input_data[0]).transform(sample_input_data[0])

    # Then
    assert transformed_df['weathersit'].iloc[2] == 'Clear'


def test_mapper_transformer(sample_input_data):
    # Given
    transformer = Mapper(config.model_config_.holiday_var, config.model_config_.holiday_mapping)

    # When
    transformed_df = transformer.fit(sample_input_data[0]).transform(sample_input_data[0])

    # Then
    # Check that the 'holiday' column values have been correctly mapped to integers
    assert transformed_df['holiday'].iloc[0] == 1  


def test_weekday_one_hot_encoder(sample_input_data):
    # Given
    encoder = WeekdayOneHotEncoder()

    # When
    transformed_df = encoder.fit(sample_input_data[0]).transform(sample_input_data[0])

    # Then
    assert 'weekday_Fri' in transformed_df.columns
    assert 'weekday_Mon' in transformed_df.columns


# def test_pipeline_prediction(sample_input_data):
#     # Given
#     X_train, X_test, y_train, y_test = train_test_split(
#         sample_input_data.drop(columns=['target']),
#         sample_input_data['target'],
#         test_size=0.2,
#         random_state=42
#     )

#     # When
#     bikeshare_pipe.fit(X_train, y_train)
#     y_pred = bikeshare_pipe.predict(X_test)

#     # Then
#     assert y_pred.shape == y_test.shape  # Ensure prediction array matches test set
#     assert accuracy_score(y_test, y_pred) > 0.5