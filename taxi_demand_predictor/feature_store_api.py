from typing import Optional
import hsfs
import hopsworks

import taxi_demand_predictor.config as config

def get_project() -> hopsworks.project.Project:
    """
    Returns a pointer to the Hopsworks project
    """
    return hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME,
        api_key_value=config.HOPSWORKS_API_KEY
    )

def get_feature_store() -> hsfs.feature_store.FeatureStore:
    """Connects to Hopsworks and returns a pointer to the feature store

    Returns:
        hsfs.feature_store.FeatureStore: pointer to the feature store
    """
    project = get_project()
    return project.get_feature_store()

def get_feature_group(
    name: str = config.FEATURE_GROUP_NAME,
    version: Optional[int] = config.FEATURE_GROUP_VERSION
    ) -> hsfs.feature_group.FeatureGroup:
    """Connects to the feature store and returns a pointer to the given
    feature group `name`

    Args:
        name (str): name of the feature group
        version (Optional[int], optional): _description_. Defaults to 1.

    Returns:
        hsfs.feature_group.FeatureGroup: pointer to the feature group
    """
    return get_feature_store().get_feature_group(
        name=name,
        version=version,
    )