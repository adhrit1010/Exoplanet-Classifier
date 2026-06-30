# Convenience targets (Linux/macOS/CI; Windows users can run the commands directly).
.PHONY: help install train train-fast predict app test lint notebook docker clean

help:
	@echo "install     Install dependencies"
	@echo "train       Full training run (model zoo + Optuna + SHAP)"
	@echo "train-fast  Quick smoke training (no slow models, few trials)"
	@echo "app         Launch the Streamlit dashboard"
	@echo "test        Run the unit tests"
	@echo "lint        flake8 on src/"
	@echo "notebook    Execute the notebook end-to-end"
	@echo "docker      Build the Docker image"

install:
	pip install -r requirements.txt

train:
	python -m src.train --trials 40

train-fast:
	python -m src.train --quick --no-slow --trials 10

predict:
	python -m src.predict --input data/KOI_Cumulative_clean.csv --output outputs/predictions/scored.csv

app:
	streamlit run app/streamlit_app.py

test:
	pytest -q

lint:
	flake8 src --max-line-length=100

notebook:
	jupyter nbconvert --to notebook --execute --inplace notebooks/01_exoplanet_classifier.ipynb

docker:
	docker build -t exoplanet-classifier .

clean:
	rm -rf outputs/plots/*.png outputs/predictions/*.csv models/*.pkl mlruns catboost_info
