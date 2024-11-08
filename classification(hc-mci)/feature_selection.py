"""
Feature Selection Script

This script performs feature selection on a dataset using various machine learning models.
It calculates feature importances and plots the results, identifying the optimal number of features
for each model based on the AUC (Area Under the Curve) metric.

The script uses the following models:
- Stochastic Gradient Descent Classifier (SGDClassifier)
- Linear Support Vector Machine Classifier (SVC)
- Multi-Layer Perceptron Classifier (MLPClassifier)
- Voting Classifier (VotingClassifier), based on the previous 3 models (SGDC, SVC and MLP)

The script assumes the following input data files:
- digimoca_results_totales.csv
- datos_participantes_totales.csv

The output of the script includes:
- CSV files containing the best features for each model
- Plots of feature importances and AUC scores for each model
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, CategoricalNB, BernoulliNB
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.inspection import permutation_importance
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PolynomialFeatures, MaxAbsScaler, Normalizer


def plot_feature_importances(fis, auc_list, name):
    """
    Plot feature importances and AUC scores.
    
    Args:
        fis (numpy.ndarray): Average feature importances.
        auc_list (list): AUC scores for different numbers of features.
        name (str): Name of the model for the output file.
    """
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel('Features')
    ax1.set_ylabel('AVG Importances', color=color)
    ax1.bar(range(1, len(auc_list) + 1), fis, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:red'
    ax2.set_ylabel('AUC', color=color)  # we already handled the x-label with ax1
    ax2.set_ylim(0, 1)

    ax2.plot(range(1, len(auc_list) + 1), auc_list, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    xmax = auc_list.index(max(auc_list)) + 1
    ymax = max(auc_list)
    text = "# Features = {}, AUC = {:.3f}".format(xmax, ymax)
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops = dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data', textcoords="axes fraction", arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax2.annotate(text, xy=(xmax, ymax), xytext=(0.94, 0.96), **kw)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig("results/feature_importances_" + name + ".png")
    plt.clf()


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

# Iterate through the estimators and perform feature selection
for model in estimators:
    model['model'].fit(X, y)
    result = permutation_importance(model['model'], X, y, scoring='roc_auc', n_repeats=100, n_jobs=-1)
    avg_fi = pd.Series(result.importances_mean, index=X.columns.tolist())  # List of average feature importances

    auc_list = []
    for i in range(1, X.shape[1] + 1):
        X_best = X[avg_fi.nlargest(i).index]
        y_pred = cross_val_predict(model['model'], X_best, y, cv=logo.split(merged, groups=ids), n_jobs=-1)
        auc_list.append(roc_auc_score(y, y_pred))

    print(model['name'], auc_list.index(max(auc_list)) + 1, max(auc_list))

    # Save the best features for each model
    avg_fi.nlargest(auc_list.index(max(auc_list)) + 1).to_csv("results/best_features_" + model['name'] + ".csv")
    plot_feature_importances(avg_fi.sort_values(ascending=False).values, auc_list, model['name'])
