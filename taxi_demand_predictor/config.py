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

FEATURE_GROUP_NAME = "time_series_hourly_feature_group"
FEATURE_GROUP_VERSION = 1

FEATURE_VIEW_NAME = "time_series_hourly_view"
FEATURE_VIEW_VERSION = 1
