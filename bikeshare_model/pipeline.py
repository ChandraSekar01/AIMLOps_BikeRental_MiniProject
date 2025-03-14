import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from bikeshare_model.config.core import config
from bikeshare_model.processing.features import WeekdayImputer, WeathersitImputer
from bikeshare_model.processing.features import Mapper
from bikeshare_model.processing.features import OutlierHandler
from bikeshare_model.processing.features import WeekdayOneHotEncoder
from bikeshare_model.processing.features import DropColumnsTransformer

bikeshare_pipe=Pipeline([
    ("weekday_imputation", WeekdayImputer()),
    ("weather_imputation", WeathersitImputer()),
    ("map_year", Mapper(config.model_config_.year_var, config.model_config_.year_mapping)),
    ("map_month", Mapper(config.model_config_.month_var, config.model_config_.month_mapping)),
    ("map_season", Mapper(config.model_config_.season_var, config.model_config_.season_mappings)),
    ("map_weathersit", Mapper(config.model_config_.weather_var, config.model_config_.weather_mappings)),
    ("map_holiday", Mapper(config.model_config_.holiday_var, config.model_config_.holiday_mapping)),
    ("map_workingday", Mapper(config.model_config_.workday_var, config.model_config_.workingday_mapping)),
    ("map_hour", Mapper(config.model_config_.hour_var, config.model_config_.hour_mapping)),
    ("outlier_handling", OutlierHandler(['temp', 'atemp', 'hum', 'windspeed'])),
    ("weekday_ohe", WeekdayOneHotEncoder()),
    ("drop_columns", DropColumnsTransformer(config.model_config_.unused_fields)),
    ("scaler", StandardScaler()),
    #  ('model_lr', LinearRegression())
    ('model_rf', RandomForestRegressor(n_estimators=config.model_config_.n_estimators, 
                                        max_depth=config.model_config_.max_depth, 
                                        random_state=config.model_config_.random_state))
                                        ])
