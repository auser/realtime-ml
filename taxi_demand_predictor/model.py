import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline
import lightgbm as lgb

def average_rides_last_4_weeks(X: pd.DataFrame) -> pd.DataFrame:
    X['average_rides_last_4_weeks'] = 0.25* (
        X[f'rides_previous_{7*24}_hour'] +  \
        X[f'rides_previous_{2*7*24}_hour'] + \
        X[f'rides_previous_{3*7*24}_hour'] + \
        X[f'rides_previous_{4*7*24}_hour']
    )
    return X

class TemporalFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X_train: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame, y=None):
        X_ = X.copy()

        X_["hour"] = X_["pickup_hour"].dt.hour
        X_["day_of_week"] = X_["pickup_hour"].dt.dayofweek

        return X_.drop(columns=["pickup_hour"])
    
def get_pipeline(**hyperparams) -> Pipeline:
    add_feature_average_rides_last_4_weeks = FunctionTransformer(
        average_rides_last_4_weeks,
        validate=False
    )

    add_temporal_features = TemporalFeatureEngineer()

    return make_pipeline(
    add_feature_average_rides_last_4_weeks,
    add_temporal_features,
    lgb.LGBMRegressor(**hyperparams)
)