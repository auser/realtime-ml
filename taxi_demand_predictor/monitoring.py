from datetime import datetime, timedelta
import pandas as pd

import taxi_demand_predictor.config as config
from taxi_demand_predictor.feature_store_api import get_feature_store, get_feature_group

def load_predictions_and_actual_values_from_store(
        from_date: datetime,
        to_date: datetime
) -> pd.DataFrame:
    """Fetch model predictions and values
    Args:
        from_date (datetime): min datetime (rounded hour) for which we want to get
        predictions
        to_date (datetime): max datetime (rounded hour) for which we want to get
        predictions
        
    Returns:
        pd.DataFrame: 3 columns:
            - `pickup_location_id`
            - `predicted_demand`
            - `actual_demand`
            - `pickup_hour`"""
    predictions_fg = get_feature_group(config.FEATURE_GROUP_MODEL_PREDICTIONS)
    actuals_fg = get_feature_group(config.FEATURE_GROUP_NAME)

    # Query join 2 feature groups by `pickup_hour` and `pickup_location_id`
    query = predictions_fg.select_all() \
            .join(actuals_fg.select_all(), on=['pickup_hour', 'pickup_location_id']) \
            .filter(predictions_fg.pickup_hour >= from_date) \
            .filter(predictions_fg.pickup_hour <= to_date)
    
    # Create feature view if it doesn't exit
    feature_store = get_feature_store()
    try:
        feature_store.create_feature_view(
            name=config.FEATURE_VIEW_MONITORING,
            version=1,
            query=query
        )
    except:
        print('Feature group dalready exists, skipping creating feature view')

    # Monitoring feature view
    monitoring_fv = feature_store.get_feature_view(
        name=config.FEATURE_VIEW_MONITORING,
        version=1
    )

    # Fetch data from the view
    # and fetch it from the last 30 days
    monitoring_df = monitoring_fv.get_batch_data(
        start_time=from_date - timedelta(days=7),
        end_time=to_date + timedelta(days=7)
    )
    monitoring_df = monitoring_df[monitoring_df.pickup_hour.between(from_date, to_date)]
    return monitoring_df