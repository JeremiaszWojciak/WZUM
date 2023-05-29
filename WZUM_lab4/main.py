import numpy as np
import pandas as pd
import seaborn as sns
import scipy
import pickle
import xgboost as xgb
import time
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn import impute
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest, VotingClassifier, StackingClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import KNeighborsClassifier


# TODO 1 ---------------------------------------------------------------------------------------------------------------

def ex_1():
    diabetes = datasets.fetch_openml(data_id=37, parser='auto')
    X, y = diabetes.data, diabetes.target
    # print(X)
    # print(y.info())
    # print(X.describe())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Missing values rejection
    # y_train = y_train.loc[(X_train['mass'] != 0.0) & (X_train['skin'] != 0.0)]
    # X_train = X_train.loc[(X_train['mass'] != 0.0) & (X_train['skin'] != 0.0)]

    # SimpleImputer mean
    # imputer_mass_mean = impute.SimpleImputer(missing_values=0.0, strategy='mean')
    # imputer_skin_mean = impute.SimpleImputer(missing_values=0.0, strategy='mean')
    # X_train[['mass']] = imputer_mass_mean.fit_transform(X_train[['mass']])
    # X_train[['skin']] = imputer_skin_mean.fit_transform(X_train[['skin']])
    # X_test[['mass']] = imputer_mass_mean.transform(X_test[['mass']])
    # X_test[['skin']] = imputer_skin_mean.transform(X_test[['skin']])

    # SimpleImputer median
    # imputer_mass_median = impute.SimpleImputer(missing_values=0.0, strategy='median')
    # imputer_skin_median = impute.SimpleImputer(missing_values=0.0, strategy='median')
    # X_train[['mass']] = imputer_mass_median.fit_transform(X_train[['mass']])
    # X_train[['skin']] = imputer_skin_median.fit_transform(X_train[['skin']])
    # X_test[['mass']] = imputer_mass_median.transform(X_test[['mass']])
    # X_test[['skin']] = imputer_skin_median.transform(X_test[['skin']])

    # SimpleImputer most frequent
    # imputer_mass_most_frequent = impute.SimpleImputer(missing_values=0.0, strategy='most_frequent')
    # imputer_skin_most_frequent = impute.SimpleImputer(missing_values=0.0, strategy='most_frequent')
    # X_train[['mass']] = imputer_mass_most_frequent.fit_transform(X_train[['mass']])
    # X_train[['skin']] = imputer_skin_most_frequent.fit_transform(X_train[['skin']])
    # X_test[['mass']] = imputer_mass_most_frequent.transform(X_test[['mass']])
    # X_test[['skin']] = imputer_skin_most_frequent.transform(X_test[['skin']])

    # KNNImputer
    imputer_mass_knn = impute.KNNImputer(n_neighbors=2, missing_values=0.0)
    imputer_skin_knn = impute.KNNImputer(n_neighbors=2, missing_values=0.0)
    X_train[['mass']] = imputer_mass_knn.fit_transform(X_train[['mass']])
    X_train[['skin']] = imputer_skin_knn.fit_transform(X_train[['skin']])
    X_test[['mass']] = imputer_mass_knn.fit_transform(X_test[['mass']])
    X_test[['skin']] = imputer_skin_knn.fit_transform(X_test[['skin']])

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


# TODO 2 ---------------------------------------------------------------------------------------------------------------
def ex_2():
    diabetes = datasets.fetch_openml(data_id=37, parser='auto')
    X, y = diabetes.data, diabetes.target

    plt.figure()
    plt.title('matplotlib boxplot')
    plt.boxplot(x=X['mass'], patch_artist=True, vert=False)

    plt.figure()
    plt.title('matplotlib histogram')
    plt.hist(x=X['mass'], bins=20)

    plt.figure()
    plt.title('seaborn boxplot')
    sns.boxplot(x=X['mass'], orient='h')

    plt.figure()
    plt.title('seaborn histogram')
    sns.histplot(data=X, x='mass', bins=20)
    plt.show()


# TODO 3 AND 4 ---------------------------------------------------------------------------------------------------------

def ex_3_4():
    diabetes = datasets.fetch_openml(data_id=37, parser='auto')
    X, y = diabetes.data, diabetes.target

    plt.figure()
    plt.scatter(X['mass'], X['plas'])
    plt.axvline(x=0)
    plt.axhline(y=0)
    plt.title('mass vs plas original')
    plt.xlabel('mass')
    plt.ylabel('plas')

    X_zscore = X.apply(scipy.stats.zscore)
    X_filtered = X[abs(X_zscore) < 3]

    plt.figure()
    plt.scatter(X_filtered['mass'], X_filtered['plas'])
    plt.axvline(x=0)
    plt.axhline(y=0)
    plt.title('mass vs plas filtered by zscore')
    plt.xlabel('mass')
    plt.ylabel('plas')
    plt.show()


# TODO 5 ---------------------------------------------------------------------------------------------------------------

def ex_5():
    diabetes = datasets.fetch_openml(data_id=37, parser='auto')
    X, y = diabetes.data, diabetes.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    X_train = X_train.values[:, [1, 5]]
    X_test = X_test.values[:, [1, 5]]

    # Original
    plt.figure()
    plt.scatter(X_test[:, 1], X_test[:, 0])
    plt.axvline(x=0)
    plt.axhline(y=0)
    plt.title('mass vs plas original')
    plt.xlabel('mass')
    plt.ylabel('plas')

    # IsolationForest
    isolation_forest = IsolationForest(random_state=42, contamination=0.05)
    isolation_forest.fit(X_train)
    outliers = isolation_forest.predict(X_test)

    X_test_inliers = X_test[outliers == 1]
    X_test_outliers = X_test[outliers == -1]

    plt.figure()
    plt.scatter(X_test_inliers[:, 1], X_test_inliers[:, 0], c='blue')
    plt.scatter(X_test_outliers[:, 1], X_test_outliers[:, 0], c='red')
    plt.axvline(x=0)
    plt.axhline(y=0)
    plt.title('mass vs plas IsolationForest outliers')
    plt.xlabel('mass')
    plt.ylabel('plas')

    # EllipticEnvelope
    elliptic_envelope = EllipticEnvelope(random_state=42, contamination=0.05)
    elliptic_envelope.fit(X_train)
    outliers = elliptic_envelope.predict(X_test)

    X_test_inliers = X_test[outliers == 1]
    X_test_outliers = X_test[outliers == -1]

    plt.figure()
    plt.scatter(X_test_inliers[:, 1], X_test_inliers[:, 0], c='blue')
    plt.scatter(X_test_outliers[:, 1], X_test_outliers[:, 0], c='red')
    plt.axvline(x=0)
    plt.axhline(y=0)
    plt.title('mass vs plas EllipticEnvelope outliers')
    plt.xlabel('mass')
    plt.ylabel('plas')
    plt.show()


# TODO 6 ---------------------------------------------------------------------------------------------------------------

def ex_6():
    pass


# TODO 7 ---------------------------------------------------------------------------------------------------------------

def ex_7():
    irises = datasets.load_iris()
    X, y = irises.data, irises.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # GridSearch SVC
    parameters = {
        'kernel': ('linear', 'rbf'),
        'C': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    }
    clf = GridSearchCV(SVC(), parameters, cv=10)
    clf.fit(X_train, y_train)
    print('GridSearch SVC best params: ', clf.best_params_)
    print('GridSearch SVC best score: ', clf.best_score_)

    plt.figure()
    pvt = pd.pivot_table(
        pd.DataFrame(clf.cv_results_),
        values='mean_test_score',
        index='param_kernel',
        columns='param_C'
    )
    ax = sns.heatmap(pvt)

    # GridSearch DecisionTreeClassifier
    parameters = {
        'criterion': ['gini', 'entropy', 'log_loss'],
        'min_samples_split': [1, 2, 3],
    }

    clf = GridSearchCV(DecisionTreeClassifier(), parameters, cv=10)
    clf.fit(X_train, y_train)
    print('GridSearch DTC best params: ', clf.best_params_)
    print('GridSearch DTC best score: ', clf.best_score_)

    plt.figure()
    pvt = pd.pivot_table(
        pd.DataFrame(clf.cv_results_),
        values='mean_test_score',
        index='param_criterion',
        columns='param_min_samples_split'
    )
    ax = sns.heatmap(pvt)
    plt.show()

    # RandomizedSearchCV SVC
    parameters = {
        'kernel': ('linear', 'rbf'),
        'C': scipy.stats.loguniform(1e0, 1e3),
    }
    clf = RandomizedSearchCV(SVC(), parameters, cv=10, n_iter=100)
    clf.fit(X_train, y_train)
    print('RandomizedSearch SVC best params: ', clf.best_params_)
    print('Randomized SVC best score: ', clf.best_score_)


# TODO 8 ---------------------------------------------------------------------------------------------------------------

def ex_8():
    irises = datasets.load_iris()
    X, y = irises.data, irises.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    parameters = {
        'kernel': ('linear', 'rbf'),
        'C': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    }
    clf = GridSearchCV(SVC(), parameters, cv=10)
    clf.fit(X_train, y_train)

    print('Prediction before saving: ', clf.best_estimator_.predict(X_test))
    print('Score before saving: ', clf.best_estimator_.score(X_test, y_test))

    pickle.dump(clf.best_estimator_, open('svc.pkl', 'wb'))
    loaded_clf = pickle.load(open('svc.pkl', 'rb'))

    print('Prediction after saving: ', loaded_clf.predict(X_test))
    print('Score after saving: ', loaded_clf.score(X_test, y_test))


# TODO 9 ---------------------------------------------------------------------------------------------------------------

def ex_9():
    irises = datasets.load_iris()
    X, y = irises.data, irises.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    svc_clf = SVC(kernel='rbf', C=7, random_state=42)
    scores = cross_val_score(svc_clf, X_train, y_train, cv=10)
    print('Scores:\n', scores)


# TODO 10 --------------------------------------------------------------------------------------------------------------

def ex_10():
    irises = datasets.load_iris()
    X, y = irises.data, irises.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.03, random_state=42, stratify=y)

    svc_clf = SVC(kernel='rbf', C=7, random_state=42, probability=True)
    svc_clf.fit(X_train, y_train)
    y_pred = svc_clf.predict_proba(X_test)
    print('Predicted class probabilities:\n', y_pred)

    plt.figure('Class probability SVC')
    x_pos = np.arange(y_pred.shape[0])
    plt.bar(x=x_pos, height=y_pred[:, 0], color='r', width=0.25)
    plt.bar(x=x_pos + 0.25, height=y_pred[:, 1], color='g', width=0.25)
    plt.bar(x=x_pos + 0.5, height=y_pred[:, 2], color='b', width=0.25)
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf1 = SVC(kernel='rbf', C=7, random_state=42, probability=True)
    clf2 = RandomForestClassifier(random_state=42)
    clf3 = DecisionTreeClassifier(random_state=42)
    clf4 = KNeighborsClassifier()

    # VotingClassifier
    eclf1 = VotingClassifier(estimators=[('svc', clf1), ('rf', clf2), ('tree', clf3), ('kn', clf4)], voting='hard')
    eclf1.fit(X_train, y_train)
    print('Voting predictions:\n', eclf1.predict(X_test))
    print('Voting score:\n', eclf1.score(X_test, y_test))

    # StackingClassifier
    eclf2 = StackingClassifier(estimators=[('svc', clf1), ('rf', clf2), ('tree', clf3), ('kn', clf4)],
                               final_estimator=LogisticRegression())
    eclf2.fit(X_train, y_train)
    print('Stacking predictions:\n', eclf2.predict(X_test))
    print('Stacking score:\n', eclf2.score(X_test, y_test))


# TODO 11 --------------------------------------------------------------------------------------------------------------

def ex_11():
    pass


# TODO 12 --------------------------------------------------------------------------------------------------------------

def ex_12():
    irises = datasets.load_iris()
    X, y = irises.data, irises.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scikit learn classifier
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    print('Feature importances: ', clf.feature_importances_)

    # XGBoots Classifier
    xgb_clf = xgb.XGBRFClassifier()
    xgb_clf.fit(X_train, y_train)
    xgb.plot_importance(xgb_clf)
    plt.show()


# TODO 13 --------------------------------------------------------------------------------------------------------------

def ex_13():
    irises = datasets.load_iris()
    X, y = irises.data, irises.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scikit learn classifier
    clf = DecisionTreeClassifier(random_state=42)
    scikit_start = time.time()
    clf.fit(X_train, y_train)
    scikit_end = time.time()

    print('Scikit score: ', clf.score(X_test, y_test))
    print('Scikit learning time:', round((scikit_end - scikit_start) * 1000, 3), 'ms')

    # XGBoots Classifier
    xgb_clf = xgb.XGBRFClassifier(random_state=42)
    xgb_start = time.time()
    xgb_clf.fit(X_train, y_train)
    xgb_end = time.time()

    print('XGBoost score: ', xgb_clf.score(X_test, y_test))
    print('XGBoost learning time:', round((xgb_end - xgb_start) * 1000, 3), 'ms')


if __name__ == '__main__':
    ex_13()
