import os
from dotenv import load_dotenv
from taxi_demand_predictor.paths import PARENT_DIR
import hopsworks

# load key-value pairs from .env file located in the parent directory
load_dotenv(PARENT_DIR / '.env')

try:
  HOPSWORKS_API_KEY = os.environ['HOPSWORKS_API_KEY']
  HOPSWORKS_PROJECT_NAME = os.environ['HOPSWORKS_PROJECT_NAME']
except:
  raise Exception('Please set HOPSWORKS_API_KEY and HOPSWORKS_PROJECT_NAME in .env file')

FEATURE_GROUP_NAME = 'time_series_hourly_feature_group'
FEATURE_GROUP_VERSION = 1
FEATURE_VIEW_NAME = 'time_series_hourly_feature_view'
FEATURE_VIEW_VERSION = 1
MODEL_NAME = "taxi_demand_predictor_next_hour"
MODEL_VERSION = 2

FEATURE_GROUP_MODEL_PREDICTIONS = 'model_predictions_feature_group'
FEATURE_VIEW_MODEL_PREDICTIONS = 'model_predictions_feature_view'
FEATURE_VIEW_MONITORING = 'predictions_vs_actuals_for_monitoring_feature_view'

# number of historical values our model needs to generate predictions
N_FEATURES = 24 * 28

# maximum Mean Absolute Error we allow our production model to have
MAX_MAE = 4.0