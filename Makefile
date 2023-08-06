.PHONE: features training inference frontend monitoring

# generates new batch of features and stores them in the feature store
features:
	poetry run python scripts/feature_pipeline.py

# trains a new model and stores it in the model registry
training:
	poetry run python scripts/training_pipeline.py

# generates predictions and stores them in the feature store
inference:
	poetry run python scripts/inference_pipeline.py

# starts the Streamlit app
frontend:
	poetry run streamlit run src/frontend.py

monitoring:
	poetry run streamlit run src/frontend_monitoring.py