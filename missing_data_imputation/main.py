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
from sklearn import metrics


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
sns.set_style("darkgrid", {"legend.frameon": True})


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


def evaluate():
    pass


def compare_models():
    pass


def feature_col_vs_metric_score(
    results_df,
    feature_col="missing_frac",
    metric_score="metric_score",
    group_col="strategy",
):
    """
    How does each model perform based on a single feature (averaged across other relevant features) wrt a metric score
    :param group_col:
    :param results_df:
    :param feature_col:
    :param metric_score:
    :return:
    """
    # TODO: show white background for legend
    return sns.lineplot(
        x=feature_col,
        y=metric_score,
        hue=group_col,
        style=group_col,
        markers=True,
        dashes=False,
        data=results_df,
    )


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

    # TODO: complete case is no imputation strategy...
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
        "knn": lambda: KNN(**strategy_kwargs),
        "mice": lambda: IterativeImputer(
            max_iter=10, sample_posterior=True, **strategy_kwargs
        ),
    }

    if columns is None:
        columns = _X.columns

    imputer = strategy_dict[strategy]()
    _X.loc[:, columns] = imputer.fit_transform(_X.loc[:, columns])

    return _X


def experiment(
    X, y, model=None, metric=None, reps=3, missing_fracs=None, impute_params=None
):

    if missing_fracs is None:
        missing_fracs = np.linspace(0.0, 0.9, 10)

    if impute_params is None:
        impute_params = [
            ("complete_case", {}),
            ("mean", {}),
            ("median", {}),
            ("most_frequent", {}),
            ("zero", {}),
            # TODO: problem if user wants to try different knn (with different k)
            #  => store strategy and some indicator!
            ("knn", {"fill_method": "mean", "k": 3}),
            ("mice", {}),
        ]

    results = {"exp_rep": [], "missing_frac": [], "strategy": [], "metric_score": []}

    print("Split into train and validation set")

    # TODO: same split for all repetitions? or resplit?
    #  Better: use CV
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, stratify=y
    )

    print("Evaluating different imputation methods w.r.t. model accuracy")

    for rep in range(reps):
        print("\n\n========== Experiment instance %s ==========" % rep)
        for missing_frac in missing_fracs:
            print(
                "\n========== Missing percentage %s%% =========="
                % int(missing_frac * 100)
            )
            for (impute_strategy, impute_param) in impute_params:
                print("========== Imputation strategy %s ==========" % impute_strategy)

                X_train_miss = delete_datapoints(X_train, frac=missing_frac)
                X_train_imputed = impute_data(
                    X_train_miss, strategy=impute_strategy, **impute_param
                )
                y_train_imputed = y_train.loc[X_train_imputed.index]

                # print("Retrain model on imputed data")
                try:
                    model.fit(X_train_imputed, y_train_imputed)
                    # make use of metric
                    y_pred = model.predict(X_test)
                    metric_score = metric(y_test, y_pred)
                except ValueError as e:
                    print(
                        "Could not train model or compute accuracy. Continue to next training instance."
                    )
                    print(e)
                    continue
                # print("Metric score: %s" % metric_score)

                results["exp_rep"].append(rep)
                results["missing_frac"].append(missing_frac)
                results["strategy"].append(impute_strategy)
                results["metric_score"].append(metric_score)

    return pd.DataFrame(results)


if __name__ == "__main__":
    print("Load data")
    X, y = load_data()
    print("Features:\n%s" % X)
    print("Labels:\n%s" % y)

    print("Standardising features")
    X = pd.DataFrame(pre_process(X, y), columns=X.columns)
    print("Features:\n%s" % X)

    missing_fracs = np.linspace(0.0, 0.9, 10)

    results_df = experiment(
        X,
        y,
        model=RandomForestClassifier(n_estimators=10, n_jobs=2),
        metric=metrics.accuracy_score,
        reps=3,
        missing_fracs=missing_fracs,
    )

    # show all results
    # complete_case should be in the end
    pd.set_option("display.max_rows", results_df.shape[0])
    pd.set_option("display.max_columns", results_df.shape[1])
    print(results_df.sort_values(by=["metric_score"], ascending=False))
    pd.reset_option("display.max_rows")
    pd.reset_option("display.max_columns")

    feature_col_vs_metric_score(results_df)
    plt.gca().set_ylim(bottom=0)
    plt.xticks(missing_fracs)
    plt.show()
