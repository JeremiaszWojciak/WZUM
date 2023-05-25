import numpy as np
import pandas as pd
import statistics
import json
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions


# TODO 1 ---------------------------------------------------------------------------------------------------------------

def zad_1():
    df = pd.read_csv('trainingdata.txt', header=None)
    print(df)

    # Statistics
    X, y = df[0].values, df[1].values

    print('Mean X np: ', np.mean(X))
    print('Mean X stat: ', statistics.mean(X))

    print('Variance X np: ', np.var(X, ddof=1))
    print('Variance X stat: ', statistics.variance(X))

    print('Mean y np: ', np.mean(y))
    print('Mean y stat: ', statistics.mean(y))

    print('Variance y np: ', np.var(y, ddof=1))
    print('Variance y stat: ', statistics.variance(y))

    print('Correlation coef np: ', np.corrcoef(X, y)[1, 0])
    print('Correlation coef stat: ', statistics.correlation(X, y))

    slope, intercept = statistics.linear_regression(X, y)
    print(f'y = {slope} * x + {intercept}')

    y_pred = slope * X + intercept
    print('R2 score: ', r2_score(y, y_pred))

    # Regression models

    X = X.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    lin_predicted = lin_reg.predict(X_test)
    print('MSE lin: ', mean_squared_error(y_test, lin_predicted))

    poly_reg = Pipeline(
        [
            ('poly', PolynomialFeatures(degree=9)),
            ('linear', LinearRegression())
        ]
    )
    poly_reg.fit(X_train, y_train)
    poly_predicted = poly_reg.predict(X_test)
    print('MSE poly: ', mean_squared_error(y_test, poly_predicted))

    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(X_train, y_train)
    tree_predicted = tree_reg.predict(X_test)
    print('MSE tree: ', mean_squared_error(y_test, tree_predicted))

    rand_reg = RandomForestRegressor()
    rand_reg.fit(X_train, y_train.ravel())
    rand_predicted = rand_reg.predict(X_test)
    print('MSE rand: ', mean_squared_error(y_test, rand_predicted))

    plt.scatter(X_test, y_test, c='red', marker='o')
    plt.scatter(X_test, lin_predicted, c='green', marker='*')
    plt.scatter(X_test, poly_predicted, c='blue', marker='.')
    plt.scatter(X_test, tree_predicted, c='orange', marker='+')
    plt.scatter(X_test, rand_predicted, c='purple', marker='v')
    plt.show()


# TODO 2 AND 3 AND 4 AND 5 AND 6 ---------------------------------------------------------------------------------------

def zad_2_3_4_5_6():
    irises = datasets.load_iris(as_frame=True)
    print(irises.frame.describe())

    X, y = irises.data, irises.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(y_train.value_counts() / y_train.count())
    print(y_test.value_counts() / y_test.count())

    plt.figure('Original')
    plt.scatter(X.values[:, 0], X.values[:, 1])
    plt.axvline(x=0)
    plt.axhline(y=0)
    plt.title('Iris sepal features')
    plt.xlabel('sepal length (cm)')
    plt.ylabel('sepal width (cm)')

    scaler_minmax = MinMaxScaler()
    scaler_minmax.fit(X_train)
    X_minmax = scaler_minmax.transform(X)

    plt.figure('MinMaxScaler')
    plt.scatter(X_minmax[:, 0], X_minmax[:, 1])
    plt.axvline(x=0)
    plt.axhline(y=0)
    plt.title('Iris sepal features')
    plt.xlabel('sepal length (cm)')
    plt.ylabel('sepal width (cm)')

    scaler_standard = StandardScaler()
    scaler_standard.fit(X_train)
    X_standard = scaler_standard.transform(X)

    plt.figure('StandardScaler')
    plt.scatter(X_standard[:, 0], X_standard[:, 1])
    plt.axvline(x=0)
    plt.axhline(y=0)
    plt.title('Iris sepal features')
    plt.xlabel('sepal length (cm)')
    plt.ylabel('sepal width (cm)')

    plt.show()


# TODO 7 ---------------------------------------------------------------------------------------------------------------

def zad_7():
    irises = datasets.load_iris(as_frame=True)

    X, y = irises.data, irises.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    plt.figure('Original')
    plt.scatter(X.values[:, 0], X.values[:, 1])
    plt.axvline(x=0)
    plt.axhline(y=0)
    plt.title('Iris sepal features')
    plt.xlabel('sepal length (cm)')
    plt.ylabel('sepal width (cm)')

    pipe_minmax = Pipeline([('min_max_scaler', MinMaxScaler())])

    pipe_minmax.fit(X_train)
    X_minmax = pipe_minmax.transform(X)

    plt.figure('MinMaxScaler')
    plt.scatter(X_minmax[:, 0], X_minmax[:, 1])
    plt.axvline(x=0)
    plt.axhline(y=0)
    plt.title('Iris sepal features')
    plt.xlabel('sepal length (cm)')
    plt.ylabel('sepal width (cm)')

    pipe_standard = Pipeline([('standard_scaler', StandardScaler())])

    pipe_standard.fit(X_train)
    X_standard = pipe_standard.transform(X)

    plt.figure('StandardScaler')
    plt.scatter(X_standard[:, 0], X_standard[:, 1])
    plt.axvline(x=0)
    plt.axhline(y=0)
    plt.title('Iris sepal features')
    plt.xlabel('sepal length (cm)')
    plt.ylabel('sepal width (cm)')

    plt.show()


# TODO 8 ---------------------------------------------------------------------------------------------------------------

def zad_8():
    irises = datasets.load_iris(as_frame=True)
    X, y = irises.data, irises.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Without pipeline
    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(X_train)
    clf_svm = SVC(random_state=42, kernel='rbf', probability=True)
    clf_svm.fit(min_max_scaler.transform(X_train), y_train)
    acc_svm = accuracy_score(y_test, clf_svm.predict(min_max_scaler.transform(X_test)))
    print(f'acc_svm: {acc_svm}')

    # With pipeline
    pipe_svm = Pipeline(
        [
            ('min_max_scaler', MinMaxScaler()),
            ('clf_svm', SVC(random_state=42, kernel='rbf', probability=True))
        ]
    )
    pipe_svm.fit(X_train, y_train)
    acc_pipe_svm = accuracy_score(y_test, pipe_svm.predict(X_test))
    print(f'acc_pipe_svm: {acc_pipe_svm}')


# TODO 9 ---------------------------------------------------------------------------------------------------------------

def zad_9():
    irises = datasets.load_iris()
    X, y = irises.data, irises.target
    X = X[:, [0, 1]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipe_svm = Pipeline(
        [
            ('min_max_scaler', MinMaxScaler()),
            ('clf_svm', SVC(random_state=42, kernel='rbf', probability=True))
        ]
    )
    pipe_svm.fit(X_train, y_train)

    # Matplotlib

    plt.figure(figsize=(10, 8))
    X1, X2 = np.meshgrid(
        np.arange(start=X_test[:, 0].min() - 1, stop=X_test[:, 0].max() + 1, step=0.01),
        np.arange(start=X_test[:, 1].min() - 1, stop=X_test[:, 1].max() + 1, step=0.01)
    )
    plt.contourf(
        X1, X2, pipe_svm.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
        alpha=0.5, cmap=ListedColormap(('red', 'green', 'blue'))
    )
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_test)):
        plt.scatter(
            X_test[y_test == j, 0], X_test[y_test == j, 1], color=ListedColormap(('red', 'green', 'blue'))(i), label=j
        )
    plt.title('Matplotlib min max scaling classifier decision boundary')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend()

    # Mlxtend

    plt.figure()
    plot_decision_regions(X_test, y_test, clf=pipe_svm, legend=2)
    plt.title('Mlxtend min max scaling classifier decision boundary')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.show()


# TODO 10 AND 11 -------------------------------------------------------------------------------------------------------

def zad_10_11():
    irises = datasets.load_iris()
    X, y = irises.data, irises.target
    X = X[:, [0, 1]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    results_dict = {}

    log_reg_clf = LogisticRegression(random_state=42)
    log_reg_clf.fit(X_train, y_train)
    acc_log_reg = accuracy_score(y_test, log_reg_clf.predict(X_test))
    print(f'acc_log_reg: {acc_log_reg}')
    results_dict['acc_log_reg'] = acc_log_reg

    svc_clf = SVC(random_state=42)
    svc_clf.fit(X_train, y_train)
    acc_svc = accuracy_score(y_test, svc_clf.predict(X_test))
    print(f'acc_svc: {acc_svc}')
    results_dict['acc_svc'] = acc_svc

    tree_clf = DecisionTreeClassifier(random_state=42)
    tree_clf.fit(X_train, y_train)
    acc_tree = accuracy_score(y_test, tree_clf.predict(X_test))
    print(f'acc_tree: {acc_tree}')
    results_dict['acc_tree'] = acc_tree

    forest_clf = RandomForestClassifier(random_state=42)
    forest_clf.fit(X_train, y_train)
    acc_forest = accuracy_score(y_test, forest_clf.predict(X_test))
    print(f'acc_forest: {acc_forest}')
    results_dict['acc_forest'] = acc_forest

    with open('results.json', 'w') as outfile:
        json.dump(results_dict, outfile)

    plt.figure()
    plot_decision_regions(X_test, y_test, clf=log_reg_clf, legend=2)
    plt.title('LogisticRegression')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')

    plt.figure()
    plot_decision_regions(X_test, y_test, clf=svc_clf, legend=2)
    plt.title('SVC')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')

    plt.figure()
    plot_decision_regions(X_test, y_test, clf=tree_clf, legend=2)
    plt.title('DecisionTreeClassifier')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')

    plt.figure()
    plot_decision_regions(X_test, y_test, clf=forest_clf, legend=2)
    plt.title('RandomForestClassifier')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')

    plt.show()


# TODO 12 --------------------------------------------------------------------------------------------------------------

def zad_12():
    irises = datasets.load_iris()
    X, y = irises.data, irises.target
    # X = X[:, [0, 1]]
    # X = X[:, [2]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    log_reg_clf = LogisticRegression(random_state=42)
    log_reg_clf.fit(X_train, y_train)
    acc_log_reg = accuracy_score(y_test, log_reg_clf.predict(X_test))
    print(f'acc_log_reg: {acc_log_reg}')

    svc_clf = SVC(random_state=42)
    svc_clf.fit(X_train, y_train)
    acc_svc = accuracy_score(y_test, svc_clf.predict(X_test))
    print(f'acc_svc: {acc_svc}')

    tree_clf = DecisionTreeClassifier(random_state=42)
    tree_clf.fit(X_train, y_train)
    acc_tree = accuracy_score(y_test, tree_clf.predict(X_test))
    print(f'acc_tree: {acc_tree}')

    forest_clf = RandomForestClassifier(random_state=42)
    forest_clf.fit(X_train, y_train)
    acc_forest = accuracy_score(y_test, forest_clf.predict(X_test))
    print(f'acc_forest: {acc_forest}')


if __name__ == '__main__':
    zad_12()
