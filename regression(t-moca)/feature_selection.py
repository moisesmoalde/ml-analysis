import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LassoLars, BayesianRidge, TweedieRegressor, SGDRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.cross_decomposition import PLSRegression, PLSCanonical
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingRegressor, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.inspection import permutation_importance
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler


def plot_feature_importances(fis, r2_list, name):
    """
    Plots a bar chart of feature importance scores and the corresponding R-squared scores.
    
    Parameters:
    fis (numpy.ndarray): Array of average feature importance scores
    r2_list (list): List of R-squared scores for different numbers of features
    name (str): Identifier for the plot file name
    """
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel('Features')
    ax1.set_ylabel('AVG Importances', color=color)
    ax1.bar(range(1, len(r2_list) + 1), fis, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('R2 score', color=color)
    ax2.set_ylim(0, 1)

    ax2.plot(range(1, len(r2_list) + 1), r2_list, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    xmax = r2_list.index(max(r2_list)) + 1
    ymax = max(r2_list)
    text = "# Features = {}, R2 = {:.3f}".format(xmax, ymax)
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops = dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data', textcoords="axes fraction", arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax2.annotate(text, xy=(xmax, ymax), xytext=(0.94, 0.96), **kw)

    fig.tight_layout()
    plt.savefig("results/feature_importances_" + name + ".png")
    plt.clf()


# Define a list of machine learning models to evaluate
estimators = [
    {'model': LinearRegression(), 'name': 'lr'},
    {'model': Ridge(), 'name': 'ri'},
    {'model': BayesianRidge(), 'name': 'bar'},
    {'model': PLSRegression(n_components=1), 'name': 'plsr'}
]

# Load the dataset and merge participant data
dataset = pd.read_csv("digimoca_results_totales.csv")
participants = pd.read_csv("datos_participantes_totales.csv")
merged = pd.merge(participants, dataset, on='id')

# Extract the feature data and target variable
ids = merged['id']
X = merged.drop(['GDS', 'id', 'gender', 'age', 'studies', 'lawton', 'MFE', 'T-MoCA', 'timestamp'], axis=1)
scaler = RobustScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=scaler.get_feature_names_out())
y = merged[['T-MoCA']].values.ravel()

# Evaluate each machine learning model
for model in estimators:
    # Fit the model and calculate permutation importance
    model['model'].fit(X, y)
    result = permutation_importance(model['model'], X, y, scoring='r2', n_repeats=100, n_jobs=-1)
    avg_fi = pd.Series(result.importances_mean, index=X.columns.tolist())

    # Calculate R-squared scores for different numbers of features
    r2_list = []
    for i in range(1, X.shape[1] + 1):
        X_best = X[avg_fi.nlargest(i).index]
        y_pred = cross_val_predict(model['model'], X_best, y, cv=LeaveOneGroupOut().split(merged, groups=ids), n_jobs=-1)
        r2_list.append(r2_score(y, y_pred))

    # Print the model name, optimal number of features, and maximum R-squared score
    print(model['name'], r2_list.index(max(r2_list)) + 1, max(r2_list))

    # Save the best features and plot the feature importances
    avg_fi.nlargest(r2_list.index(max(r2_list)) + 1).to_csv("results/best_features_" + model['name'] + ".csv")
    plot_feature_importances(avg_fi.sort_values(ascending=False).values, r2_list, model['name'])
