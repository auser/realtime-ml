import zipfile 
from datetime import datetime, timedelta

import requests
import numpy as np
import pandas as pd
import streamlit as st
import geopandas as gpd
import pydeck as pdk

from taxi_demand_predictor.inference import (
    load_predictions_from_store,
    load_batch_of_features_from_store
)
from taxi_demand_predictor.paths import DATA_DIR
from taxi_demand_predictor.plot import plot_one_sample

st.set_page_config(layout="wide")