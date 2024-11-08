# Machine Learning Model Optimization and Evaluation

This repository contains two folders: `classification(hc-mci)` contains the code and data for classification of participants between HC and MCI, and `regression(t-moca)` contains the code and data for predicting the T-MoCA score of the participants.

In each folder, there are two Python scripts that perform feature selection optimization and model fitting/prediction on a dataset. Additionally, there is a Jupyter Notebook that documents and contains all the code from both scripts `feature_selection.py` and `fit_predict.py`.

## Dependencies
The scripts requires the following Python libraries:
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

## Configuration
All scripts allow you to customize the ML models to be evaluated by modifying the `estimators` list. Each model is represented as a dictionary with the following keys:
- `model`: An instance of the machine learning model
- `name`: A unique identifier for the model

## Usage Guidelines

### Running the Feature Selection
```bash
# Inside ./classification(hc-mci) or ./regression(t-moca)
python feature_selection.py
```
- Ensure you have the required dependencies installed
- Ensure input CSV files are in the working directory (`digimoca_results_totales.csv` and `datos_participantes_totales.csv`)
- Creates a `results` directory if it doesn't exist
- Outputs feature importance plots and best features CSV files

### Running the Model Prediction
```bash
# Inside ./classification(hc-mci) or ./regression(t-moca)
python fit_predict.py
```
- Requires the feature selection step to be completed first
- Uses the best features CSV files from the feature selection step
- Generates the scatter plot (regression) or ROC curve (classification)
- For each model, calculates and displays the R² and RMSE (regression) or the AUC (classification)

## Feature Selection Optimization

The `feature_selection.py` script performs feature selection optimization on a dataset using various machine learning models.

### Features
- Supports a variety of machine learning models for feature selection optimization
- Calculates permutation importance of each feature to determine their relative importance
- Evaluates model performance (R² for regression, AUC for classification) as the number of features is increased
- Identifies the optimal number of features for each model
- Saves the best features for each model in separate CSV files
- Generates a plot for each model, visualizing the feature importances and R²/AUC scores

### Output
The script generates the following output:
1. CSV files containing the best features for each model, saved in the `results/` directory with the naming convention `best_features_<model_name>.csv`.
2. PNG images showing the feature importance plots for each model, saved in the `results/` directory with the naming convention `feature_importances_<model_name>.png`.


## Model Fitting and Prediction

The `fit_predict.py` script is designed to perform model fitting and prediction on a dataset, using the best features identified in the previous feature selection optimization script.

1. In the regression case, it generates a scatter plot of the predicted vs. true target values for 4 selected ML models.
2. In the classification case, it generates a ROC curve plot for each ML model.

### Features
- Loads the best features for each machine learning model from the previous feature selection optimization script
- Performs cross-validation predictions for each model using the best features from `results/best_features_<model_name>.csv`
- **REGRESSION**
    - Generates a 2x2 scatter plot of the predicted vs. true target values for all the models
    - Calculates and displays the R² score and Root Mean Squared Error (RMSE) for each model
- **CLASSIFICATION**
    - Generates a ROC curve plot for all models
    - Calculates and displays the Area Under the Curve (AUC)

### Output
The script generates the following output:
1. **REGRESSION:** A PNG image showing the scatter plot of the predicted vs. true target values for 4 selected models, saved in `results/scatter_plot_<model_name_1>_<model_name_2>_..._.png`.
2. **CLASSIFICATION:** A PNG image showing the ROC curve plot for all models, saved in `results/roc_<model_name_1>_<model_name_2>_..._.png`.
