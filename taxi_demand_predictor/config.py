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

def get_feature_group(feature_store):
  feature_group = feature_store.get_or_create_feature_group(
    name=FEATURE_GROUP_NAME,
    version=FEATURE_GROUP_VERSION,
    description="Time-series data at hourly frequency",
    primary_key=['pickup_location_id', 'pickup_hour'],
    event_time='pickup_hour',
  )
  return feature_group