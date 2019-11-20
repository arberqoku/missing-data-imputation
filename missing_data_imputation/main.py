"""Sandbox for notebook"""
# base imports
import os
import time

# exploratory data analysis
import missingno, pandas_profiling

# data processing
import numpy as np
import pandas as pd

# datasets/training/imputation
import sklearn
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


# imputation
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# import impyute, fancyimpute, autoimpute, missingpy, datawig
from fancyimpute import KNN

# visualisation
from matplotlib import pyplot as plt
import seaborn as sns

# plot styling
sns.set_context("talk")
sns.set_style("darkgrid")


def load_data():
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["label"] = iris.target
    # df["label_name"] = [iris.target_names[it] for it in iris.target]
    return df.iloc[:, :-1], df.iloc[:, -1]


# generate exploratory data analysis report as html file?
def eda_report():
    pass


# maybe scale data
def pre_process(X, y):
    scaler = StandardScaler()
    return scaler.fit_transform(X, y)


# train_test_split
def split_data(X, y, test_size=0.25, **kwargs):
    return train_test_split(X, y, test_size=test_size, stratify=y, **kwargs)


def train(X, y):
    clf = RandomForestClassifier(n_estimators=10, n_jobs=2)
    clf.fit(X, y)
    return clf


def evaluate():
    pass


def plot():
    pass


def compare_models():
    pass


# introduce missingness
def delete_datapoints(X, columns=None, frac=0.1):
    _X = X.copy(deep=True)

    if columns is None:
        columns = _X.columns

    for col in columns:
        nan_idx = _X.sample(frac=frac).index
        _X.loc[nan_idx, col] = np.NaN

    return _X


def impute_data(X, columns=None, strategy="mean", **strategy_kwargs):
    _X = X.copy(deep=True)

    if strategy == "complete_case":
        return _X.dropna()

    strategy_dict = {
        # use lambda to avoid premature initialisation
        "mean": lambda: SimpleImputer(missing_values=np.NaN, strategy="mean"),
        "median": lambda: SimpleImputer(missing_values=np.NaN, strategy="median"),
        "most_frequent": lambda: SimpleImputer(
            missing_values=np.NaN, strategy="most_frequent"
        ),
        "zero": lambda: SimpleImputer(
            missing_values=np.NaN, strategy="constant", fill_value=0
        ),
        "hot-deck": lambda: None,
        "knn": lambda: KNN(k=3),
        "mice": lambda: IterativeImputer(
            max_iter=10, sample_posterior=True, **strategy_kwargs
        ),
    }

    if columns is None:
        columns = _X.columns

    imputer = strategy_dict[strategy]()
    _X.loc[:, columns] = imputer.fit_transform(_X.loc[:, columns])

    return _X


if __name__ == "__main__":
    print("Load data")
    X, y = load_data()
    print("Features:\n%s" % X)
    print("Labels:\n%s" % y)

    print("Standardising features")
    X = pd.DataFrame(pre_process(X, y), columns=X.columns)
    print("Features:\n%s" % X)

    print("Split data into train and test")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.25)

    print("Train random forest classifier")
    clf = train(X_train, y_train)

    print("Predict test data")
    y_pred = clf.predict(X_test)
    print("Model accuracy: %s" % clf.score(X_test, y_test))

    missing_fracs = np.linspace(0.1, 0.9, 9)
    impute_params = [
        ("complete_case", {}),
        ("mean", {}),
        ("median", {}),
        ("most_frequent", {}),
        ("zero", {}),
        ("knn", {"fill_method": "mean", "k": 3}),
        ("mice", {}),
    ]

    results = {"missing_frac": [], "strategy": [], "accuracy": []}

    for missing_frac in missing_fracs:
        for (impute_strategy, impute_param) in impute_params:
            print(
                "Introduce %s%% missingness in every column" % int(missing_frac * 100)
            )
            X_train_miss = delete_datapoints(X_train, frac=missing_frac)

            print("Impute missing data using %s strategy" % impute_strategy)
            X_train_imputed = impute_data(
                X_train_miss, strategy=impute_strategy, **impute_param
            )
            y_train_imputed = y_train.loc[X_train_imputed.index]

            print("Retrain random forest classifier on imputed data")
            try:
                clf_imputed = train(X_train_imputed, y_train_imputed)
                clf_accuracy = clf_imputed.score(X_test, y_test)
            except ValueError as e:
                print(e)
                clf_accuracy = 0
            print("Model accuracy: %s" % clf_accuracy)

            results["missing_frac"].append(missing_frac)
            results["strategy"].append(impute_strategy)
            results["accuracy"].append(clf_accuracy)

    results_df = pd.DataFrame(results)

    # show all results
    # complete_case should be in the end
    pd.set_option("display.max_rows", results_df.shape[0])
    pd.set_option("display.max_columns", results_df.shape[1])
    print(results_df.sort_values(by=["accuracy"], ascending=False))
    pd.reset_option("display.max_rows")
    pd.reset_option("display.max_columns")
