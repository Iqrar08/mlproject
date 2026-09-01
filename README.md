# Student Performance Prediction (mlproject)

An end-to-end machine learning pipeline that predicts a student's **math score** from demographic and academic attributes, served through a Flask web application.

## Overview

This project trains and evaluates multiple regression models on a student performance dataset, selects the best-performing model, and exposes it through a Flask web app where a user can submit a student's profile (gender, ethnicity group, parental education, lunch type, test preparation status, reading score, and writing score) and receive a predicted math score.

The codebase follows a modular ML project structure: data ingestion, data transformation, model training, and prediction are separated into distinct components under `src/`, with custom logging and exception handling used throughout.

## Key Features

- Modular data ingestion, transformation, and model-training pipeline (`src/components/`)
- Train/test split performed during ingestion (80/20, `random_state=42`)
- Preprocessing pipeline using `ColumnTransformer` with separate numerical and categorical branches (imputation, one-hot encoding, scaling)
- Hyperparameter tuning via `GridSearchCV` (`cv=3`) across 8 candidate regression models
- Automatic selection of the best model based on test R² score
- Model and preprocessor persistence to disk via `pickle` (`src/utils.py`)
- Flask web application (`application.py`) with an HTML form for entering student attributes and viewing the predicted score
- Custom exception handling (`src/exception.py`) and file-based logging (`src/logger.py`)
- Dockerfile for containerized deployment
- Exploratory Data Analysis notebook (`notebook/EDA_student_performance.ipynb`)

## Machine Learning Workflow

1. **Data Ingestion** (`src/components/data_ingestion.py`): Reads `notebook/data/stud.csv`, saves a raw copy to `artifacts/raw.csv`, splits it into train/test sets (80/20), and writes `artifacts/train.csv` and `artifacts/test.csv`.
2. **Data Transformation** (`src/components/data_transformation.py`): Builds a `ColumnTransformer` preprocessing pipeline — median imputation + `StandardScaler` for numerical columns (`reading_score`, `writing_score`), and most-frequent imputation + `OneHotEncoder` + `StandardScaler(with_mean=False)` for categorical columns (`gender`, `race_ethnicity`, `parental_level_of_education`, `lunch`, `test_preparation_course`). The target column is `math_score`. The fitted preprocessor is saved to `artifacts/preprocessor.pkl`.
3. **Model Training** (`src/components/model_trainer.py`): Trains and tunes 8 regressors (see [Models](#models)) using `GridSearchCV`, evaluates each on the test set with R², and saves the best-performing model to `artifacts/model.pkl`. Training raises a `CustomException` if the best model's R² score is below `0.6`.
4. **Prediction Pipeline** (`src/pipeline/predict_pipeline.py`): Wraps user input into a `CustomData` object, converts it to a DataFrame, loads the saved preprocessor and model from `artifacts/`, and returns a prediction.
5. **Web Application** (`application.py`): Collects form input, invokes the prediction pipeline, and renders the predicted math score back to the user.

The full ingestion → transformation → training pipeline can be run end-to-end via `src/components/data_ingestion.py`'s `__main__` block, which chains `DataIngestion`, `Datatranformation`, and `ModelTrainer` in sequence.

> Note: `src/pipeline/train_pipeline.py` exists in the repository but is currently empty.

## Dataset

- **Source file**: `notebook/data/stud.csv` (1000 rows)
- **Columns**: `gender`, `race_ethnicity`, `parental_level_of_education`, `lunch`, `test_preparation_course`, `math_score`, `reading_score`, `writing_score`
- **Target variable**: `math_score`
- **Features**: `gender`, `race_ethnicity`, `parental_level_of_education`, `lunch`, `test_preparation_course` (categorical); `reading_score`, `writing_score` (numerical)

An exploratory data analysis notebook is available at `notebook/EDA_student_performance.ipynb`.

## Models

The following regressors are trained and tuned in `src/components/model_trainer.py`, each with its own `GridSearchCV` parameter grid:

| Model | Key tuned hyperparameters |
|---|---|
| Random Forest Regressor | `n_estimators` |
| Decision Tree Regressor | `criterion` |
| Gradient Boosting Regressor | `learning_rate`, `subsample`, `n_estimators` |
| Linear Regression | — (no grid) |
| K-Neighbors Regressor | `n_neighbors` |
| XGBoost Regressor | `learning_rate`, `n_estimators` |
| CatBoost Regressor | `learning_rate`, `depth`, `iterations` |
| AdaBoost Regressor | `learning_rate`, `n_estimators` |

The model with the highest test R² score is selected as the final model and persisted to `artifacts/model.pkl`.

## Model Evaluation

Models are compared using **R² score** on the held-out test set (`src/utils.py`, `evaluate_models` function). The trained model is rejected (via `CustomException`) if its R² score falls below `0.6`.

The repository does not include a record of the final selected model's specific R² value, so no metric is reported here to avoid fabrication. Running the pipeline locally (see [Usage](#usage)) will print the achieved R² score.

## Project Architecture

```
stud.csv (raw data)
      │
      ▼
DataIngestion ──► artifacts/raw.csv, train.csv, test.csv
      │
      ▼
Datatranformation ──► artifacts/preprocessor.pkl
      │
      ▼
ModelTrainer ──► artifacts/model.pkl
      │
      ▼
PredictPipeline (loads model.pkl + preprocessor.pkl)
      │
      ▼
Flask app (application.py) ──► templates/index.html, home.html
```

`application.py` depends on `src/pipeline/predict_pipeline.py`, which in turn depends on the artifacts produced by the ingestion → transformation → training chain. Cross-cutting concerns (logging, custom exceptions) are used by every component via `src/logger.py` and `src/exception.py`.

## Project Structure

```
mlproject/
├── application.py                     # Flask app entry point
├── Dockerfile
├── requirements.txt
├── setup.py
├── artifacts/
│   ├── raw.csv
│   ├── train.csv
│   ├── test.csv
│   ├── preprocessor.pkl
│   └── model.pkl
├── notebook/
│   ├── EDA_student_performance.ipynb
│   └── data/
│       └── stud.csv
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/
│   │   ├── predict_pipeline.py
│   │   └── train_pipeline.py          # currently empty
│   ├── mlproject/
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
└── templates/
    ├── index.html
    └── home.html
```

`catboost_info/` is generated automatically by CatBoost during training and is not part of the application source.

## Tech Stack

- **Language**: Python
- **Web framework**: Flask
- **ML/Data libraries**: scikit-learn, pandas, numpy, XGBoost, CatBoost, seaborn, matplotlib
- **Serialization**: pickle, dill
- **WSGI server**: gunicorn (listed in `requirements.txt`)
- **Containerization**: Docker

## Installation

```bash
git clone https://github.com/Iqrar08/mlproject.git
cd mlproject
pip install -r requirements.txt
```

`requirements.txt` includes `-e .`, which installs the local `src` package in editable mode using `setup.py`.

## Usage

**Run the full training pipeline** (data ingestion → transformation → model training):

```bash
python src/components/data_ingestion.py
```

This regenerates `artifacts/raw.csv`, `artifacts/train.csv`, `artifacts/test.csv`, `artifacts/preprocessor.pkl`, and `artifacts/model.pkl`, and prints the resulting test R² score.

**Run the web application:**

```bash
python application.py
```

The app starts on `http://0.0.0.0:5000`.

## Web Application / API

Defined in `application.py`:

| Route | Methods | Description |
|---|---|---|
| `/` | GET | Renders `index.html` (landing page) |
| `/predictdata` | GET | Renders `home.html` (input form) |
| `/predictdata` | POST | Reads form fields (`gender`, `ethnicity`, `parental_level_of_education`, `lunch`, `test_preparation_course`, `reading_score`, `writing_score`), runs `PredictPipeline.predict()`, and re-renders `home.html` with the predicted math score |

## Docker

Build and run using the provided `Dockerfile`:

```bash
docker build -t mlproject .
docker run -p 5000:5000 mlproject
```

The container installs dependencies from `requirements.txt` and runs `python application.py` on start.

## Environment Variables

No environment variables are defined or required by the current codebase.

## Screenshots / Demo

The prediction UI (served from `templates/home.html`):

**Header**
![Header](screenshots/header.png)

**Hero section**
![Hero section](screenshots/hero.png)

**Feature highlight**
![Feature highlight](screenshots/feature-highlight.png)

**Prediction form**
![Prediction form](screenshots/prediction-form.png)

## Limitations

- `src/pipeline/train_pipeline.py` is present but empty; the training workflow currently must be triggered via `src/components/data_ingestion.py`.
- The final model's evaluation metric is not stored in the repository; it is only printed at training time.
- `application.py` runs Flask's built-in development server (`app.run(...)`), not a production WSGI server, though `gunicorn` is listed as a dependency.
- No automated tests are present in the repository.

## Future Improvements

*(Suggestions below are not implemented in the current codebase.)*

- Implement `src/pipeline/train_pipeline.py` as a callable training entry point
- Add a `gunicorn`-based production run command / Procfile
- Persist evaluation metrics (e.g., to a report file or logging output) for traceability
- Add unit/integration tests
- Add CI configuration for automated linting and testing

## Author

**Iqrar** — `iqrara629@gmail.com` (from `setup.py`)
