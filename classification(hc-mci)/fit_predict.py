"""
Model Fitting and Prediction Script

This script loads a dataset, preprocesses the data, and trains several machine learning models to predict a target variable.
It then generates ROC (Receiver Operating Characteristic) curves for each model using the cross-validated predictions.

The script assumes the following input data files:
- digimoca_results_totales.csv
- datos_participantes_totales.csv
- The best features for each model, saved in the "results" directory during the feature selection process.

The output of the script includes:
- A ROC curve plot saved in the results/ directory with the naming convention "results/roc_<model_name_1>_<model_name_2>_....png"
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, BayesianRidge, TweedieRegressor, SGDClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, CategoricalNB, BernoulliNB
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cross_decomposition import PLSRegression

from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import classification_report, RocCurveDisplay
from sklearn.preprocessing import RobustScaler


# Define the estimators (machine learning models) to be used
estimators = [
    {'model': SGDClassifier(loss='log_loss'), 'name': 'logit'},
    {'model': SVC(kernel='linear', probability=True), 'name': 'svm'},
    {'model': MLPClassifier(max_iter=100000), 'name': 'mlp'},
    {'model': VotingClassifier(estimators=[('sgdc', SGDClassifier(loss='log_loss')),
                                          ('lsvc', SVC(kernel='linear', probability=True)),
                                          ('mlpc', MLPClassifier(max_iter=100000))], voting='soft'),
                                'name': 'vote'}
]

# Load the input data
dataset = pd.read_csv("digimoca_results_totales.csv")
participants = pd.read_csv("datos_participantes_totales.csv")
merged = pd.merge(participants, dataset, on='id')

ids = merged['id']
logo = LeaveOneGroupOut()

# Prepare the feature matrix and target variable
X = merged.drop(['GDS', 'id', 'gender', 'age', 'studies', 'lawton', 'MFE', 'T-MoCA', 'timestamp'], axis=1)
scaler = RobustScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=scaler.get_feature_names_out())
y = merged['GDS'].transform(lambda gds: (0 if gds < 3 else 1)).values

# Create a figure and axis object for the ROC curve plot
fig, ax = plt.subplots()

# Iterate through the estimators and generate ROC curves
for model in estimators:
    # Load the best features for the current model
    best_features = pd.read_csv("results/best_features_" + model['name'] + ".csv", index_col=0)

    # Select the best features from the feature matrix
    X_best = X[best_features.index]

    # Obtain cross-validated prediction probabilities using the current model
    y_pred = cross_val_predict(model['model'], X_best, y, cv=logo.split(merged, groups=ids), n_jobs=-1, method='predict_proba')

    # Plot the ROC curve for the current model
    RocCurveDisplay.from_predictions(y, y_pred[:, 1], name=model['name'], ax=ax, alpha=0.8, plot_chance_level=(model['name'] == 'vr'))
    ax = plt.gca()

# Set the axis labels for the ROC curve plot
ax.set(
    xlabel="False Positive Rate",
    ylabel="True Positive Rate"
)

# Save the ROC curve plot
plt.savefig(f"results/roc_{'_'.join([model['name'] for model in estimators])}.png")
