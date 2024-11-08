import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model		import LinearRegression, Ridge, ElasticNet, BayesianRidge, TweedieRegressor
from sklearn.svm				import SVR
from sklearn.tree				import DecisionTreeRegressor
from sklearn.neighbors			import KNeighborsRegressor
from sklearn.ensemble			import RandomForestRegressor, AdaBoostRegressor, VotingRegressor, ExtraTreesRegressor
from sklearn.neural_network		import MLPRegressor
from sklearn.cross_decomposition import PLSRegression

from sklearn.model_selection	import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics			import mean_squared_error, r2_score
from sklearn.preprocessing		import RobustScaler


def plot_regression_results(ax, y_true, y_pred, title, r2, rmse):
    """
    Plots a scatter plot of the predicted vs true target values.
    
    Parameters:
    ax (matplotlib.axes.Axes): The axis object to plot on.
    y_true (numpy.ndarray): The true target values.
    y_pred (numpy.ndarray): The predicted target values.
    title (str): The title of the plot.
    r2 (float): The R-squared score.
    rmse (float): The root mean squared error.
    """
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "--r", linewidth=2)
    ax.scatter(y_true, y_pred, alpha=0.2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))
    ax.set_xlim([y_true.min()-0.5, y_true.max()+0.5])
    ax.set_ylim([y_true.min()-0.5, y_true.max()+0.5])
    ax.set_xlabel("T-MoCA")
    ax.set_ylabel("Predicted")
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False, edgecolor="none", linewidth=0)
    scores = (r"$R^2={:.2f}$" + "\n" + r"$RMSE={:.2f}$").format(r2, rmse)
    ax.legend([extra], [scores], loc="upper left")
    ax.set_title(title)

def scatter_plot(y, y_pred_list, estimators, file_name="scatter_plot.png"):
    """
    Creates a 2x2 scatter plot of the predicted vs true target values for multiple models.
    
    Parameters:
    y (numpy.ndarray): The true target values.
    y_pred_list (list): A list of predicted target values for each model.
    estimators (list): A list of dictionaries containing the model information.
    file_name (str): The name of the output file.
    """
    fig, axs = plt.subplots(2, 2, figsize=(9, 7))
    axs = np.ravel(axs)

    for ax, est, y_pred in zip(axs, estimators, y_pred_list):
        name = type(est['model']).__name__
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))

        plot_regression_results(ax, y, y_pred, name, r2, rmse)

    plt.suptitle("LOGO Cross-Validation Scatter Plot")
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(file_name)


# Define a list of 4 machine learning models to evaluate
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

y_pred_list = []

# Iterate through the models, load the best features, and make predictions
for model in estimators:
	best_features = pd.read_csv("results/best_features_" + model['name'] + ".csv", index_col = 0)

	X_best = X[best_features.index]

	y_pred = cross_val_predict(model['model'], X_best, y, cv = LeaveOneGroupOut().split(merged, groups=ids), n_jobs = -1)

	y_pred_list.append(y_pred)

# Generate the scatter plot and save it to the 'results/' directory
file_name = f"results/scatter_plot_{'_'.join([model['name'] for model in estimators])}.png"
scatter_plot(y, y_pred_list, estimators, file_name = file_name)
