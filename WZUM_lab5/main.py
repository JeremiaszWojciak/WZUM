import numpy as np
import pandas as pd
import random
import missingno as msno
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import impute
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

pd.options.display.max_columns = None


# TODO 1 - 11 ----------------------------------------------------------------------------------------------------------

def random_cabin(clas):
    if clas == 1.0:
        deck = random.choices(['C', 'B', 'D', 'E', 'A', 'T'],
                              weights=[0.346405, 0.241830, 0.169935, 0.130719, 0.104575, 0.006536])
    elif clas == 2.0:
        deck = random.choices(['D', 'E', 'F'], weights=[0.333333, 0.222222, 0.444444])
    else:
        deck = random.choices(['E', 'F', 'G'], weights=[0.125, 0.750, 0.125])

    number = random.randint(1, 130)

    return deck[0] + str(number)


def fill_cabin(x):
    cabin_nan = x['cabin'].isna()
    fills = []
    clas_list = list(x.loc[cabin_nan, 'pclass'])
    for clas in clas_list:
        cabin = random_cabin(clas)
        fills.append(cabin)
    x.loc[cabin_nan, 'cabin'] = fills
    return x


def fill_age(x):
    age_nan = x['age'].isna()
    fills = np.random.normal(x['age'].mean(), x['age'].std(), age_nan.sum())
    fills = [item if item >= 0. else np.random.ranf() for item in fills]
    x.loc[age_nan, 'age'] = fills
    return x


def zad_1_11():
    titanic = datasets.fetch_openml(name='Titanic', version=1, parser='auto')
    X, y = titanic.data, titanic.target
    # print(X)
    # print(y)
    # print(X.info())
    # print(X.describe())
    # print(y.info())
    # print(y.describe())

    # Delete columns 'boat', 'body' and 'home.dest'
    X = X.drop(['boat', 'body', 'home.dest'], axis=1)

    # Divide dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

    # Random predictions
    y_random = random.choices(population=['0', '1'], k=len(y_test))
    print('Random predictions report:\n', metrics.classification_report(y_test, y_random))

    # Number of missing values
    print('Number of missing values:\n', X.isna().sum(axis=0))

    # Visualisation of missing values
    # msno.matrix(X)
    # plt.show()

    # Fill embarked
    imputer_embarked = impute.SimpleImputer(missing_values=np.NaN, strategy='most_frequent')
    X_train[['embarked']] = imputer_embarked.fit_transform(X_train[['embarked']])
    X_test[['embarked']] = imputer_embarked.transform(X_test[['embarked']])

    # Fill cabin
    X_train = fill_cabin(X_train)
    X_test = fill_cabin(X_test)

    # Plot age histogram
    # X_temp = X_train.copy()
    # X_temp['age'] = X_temp['age'].dropna()
    # X_temp['age'].hist()
    # plt.show()

    # Fill age
    X_train = X_train.groupby('pclass', group_keys=False).apply(fill_age)
    X_test = X_test.groupby('pclass', group_keys=False).apply(fill_age)

    # Fill fare
    imputer_fare = impute.KNNImputer(missing_values=np.NaN, n_neighbors=3)
    X_train[['fare']] = imputer_fare.fit_transform(X_train[['fare']])
    X_test[['fare']] = imputer_fare.transform(X_test[['fare']])

    # Encode values
    enc = OrdinalEncoder()
    X_train[['name', 'sex', 'ticket', 'cabin', 'embarked']] = enc.fit_transform(
        X_train[['name', 'sex', 'ticket', 'cabin', 'embarked']])
    X_test[['name', 'sex', 'ticket', 'cabin', 'embarked']] = enc.fit_transform(
        X_test[['name', 'sex', 'ticket', 'cabin', 'embarked']])

    # Train a classifier
    clf = SVC()
    cv = cross_val_score(clf, X_train, y_train, cv=5)
    print('Basic classifier cv:', cv)
    # clf.fit(X_train, y_train)
    # print(clf.score(X_test, y_test))

    # Relationship between features and survival
    X_combined = pd.concat([X_train, y_train.astype(float)], axis=1)
    print('Pclass vs survival:\n', X_combined.groupby('pclass', as_index=False).survived.mean())
    print('Sex vs survival:\n', X_combined.groupby('sex', as_index=False).survived.mean())
    print('Embarked vs survival:\n', X_combined.groupby('embarked', as_index=False).survived.mean())
    print('Parch vs survival:\n', X_combined.groupby('parch', as_index=False).survived.mean())
    print('Sibsp vs survival:\n', X_combined.groupby('sibsp', as_index=False).survived.mean())

    # plt.figure('Pclass vs survival')
    # sns.barplot(x='pclass', y='survived', data=X_combined.groupby('pclass', as_index=False).survived.mean())
    # plt.figure('Sex vs survival')
    # sns.barplot(x='sex', y='survived', data=X_combined.groupby('sex', as_index=False).survived.mean())
    # plt.figure('Embarked vs survival')
    # sns.barplot(x='embarked', y='survived', data=X_combined.groupby('embarked', as_index=False).survived.mean())
    # plt.figure('Parch vs survival')
    # sns.barplot(x='parch', y='survived', data=X_combined.groupby('parch', as_index=False).survived.mean())
    # plt.figure('Sibsp vs survival')
    # sns.barplot(x='sibsp', y='survived', data=X_combined.groupby('sibsp', as_index=False).survived.mean())
    # plt.show()

    # Features correlation
    # plt.figure(figsize=(13, 10))
    # sns.heatmap(X_combined.corr(), annot=True, cmap="coolwarm")
    # plt.show()
    #
    # sns.pairplot(X_combined, vars=['pclass', 'age', 'sex', 'fare'], hue='survived')
    # plt.show()

    # Feature extraction


# TODO FINAL SOLUTION --------------------------------------------------------------------------------------------------

def zad_final():
    titanic = datasets.fetch_openml(name='Titanic', version=1, parser='auto')
    X, y = titanic.data, titanic.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    X_train = pd.concat([X_train, y_train.astype(int)], axis=1)
    X_test = pd.concat([X_test, y_test.astype(int)], axis=1)
    X_all = [X_train, X_test]

    # ------------------------------------------------------ Name ------------------------------------------------------

    for dataset in X_all:
        dataset['title'] = dataset['name'].str.extract(' ([A-Za-z]+)\.')

    # print(pd.crosstab(X_train['title'], X_train['sex']))

    for dataset in X_all:
        dataset['title'] = dataset['title'].replace(
            ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')
        dataset['title'] = dataset['title'].replace('Mlle', 'Miss')
        dataset['title'] = dataset['title'].replace('Ms', 'Miss')
        dataset['title'] = dataset['title'].replace('Mme', 'Mrs')

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Other": 5}
    for dataset in X_all:
        dataset['title'] = dataset['title'].map(title_mapping)
        dataset['title'] = dataset['title'].fillna(0)

    # ------------------------------------------------------ Sex -------------------------------------------------------

    for dataset in X_all:
        dataset['sex'] = dataset['sex'].map({'female': 1, 'male': 0}).astype(int)

    # ---------------------------------------------------- Embarked ----------------------------------------------------

    # print(X_train.embarked.unique())
    # print(X_train.embarked.value_counts())
    for dataset in X_all:
        dataset['embarked'] = dataset['embarked'].fillna('S')
    for dataset in X_all:
        dataset['embarked'] = dataset['embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    # ------------------------------------------------------ Age -------------------------------------------------------

    for dataset in X_all:
        age_avg = dataset['age'].mean()
        age_std = dataset['age'].std()
        age_null_count = dataset['age'].isnull().sum()
        age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
        dataset.loc[dataset['age'].isna(), 'age'] = age_null_random_list
        dataset['age'] = dataset['age'].astype(int)

    X_train['age_band'] = pd.cut(X_train['age'], 5)
    # print(X_train[['age_band', 'survived']].groupby(['age_band'], as_index=False).mean())

    for dataset in X_all:
        dataset.loc[dataset['age'] <= 16, 'age'] = 0
        dataset.loc[(dataset['age'] > 16) & (dataset['age'] <= 32), 'age'] = 1
        dataset.loc[(dataset['age'] > 32) & (dataset['age'] <= 48), 'age'] = 2
        dataset.loc[(dataset['age'] > 48) & (dataset['age'] <= 64), 'age'] = 3
        dataset.loc[dataset['age'] > 64, 'age'] = 4

    # ------------------------------------------------------ Fare ------------------------------------------------------

    for dataset in X_all:
        dataset['fare'] = dataset['fare'].fillna(X_train['fare'].median())  # Dlaczego z X_train a nie dataset?????

    X_train['fare_band'] = pd.qcut(X_train['fare'], 4)
    # print(X_train[['fare_band', 'survived']].groupby(['fare_band'], as_index=False).mean())

    for dataset in X_all:
        dataset.loc[dataset['fare'] <= 7.9, 'fare'] = 0
        dataset.loc[(dataset['fare'] > 7.9) & (dataset['fare'] <= 13.9), 'fare'] = 1
        dataset.loc[(dataset['fare'] > 13.9) & (dataset['fare'] <= 31), 'fare'] = 2
        dataset.loc[dataset['fare'] > 31, 'fare'] = 3
        dataset['fare'] = dataset['fare'].astype(int)

    # -------------------------------------------------- Sibsp & Parch -------------------------------------------------

    for dataset in X_all:
        dataset['family_size'] = dataset['sibsp'] + dataset['parch'] + 1

    # print(X_train[['family_size', 'survived']].groupby(['family_size'], as_index=False).mean())

    for dataset in X_all:
        dataset['is_alone'] = 0
        dataset.loc[dataset['family_size'] == 1, 'is_alone'] = 1

    # print(X_train[['is_alone', 'survived']].groupby(['is_alone'], as_index=False).mean())

    # ------------------------------------------------ Feature selection -----------------------------------------------

    features_drop = ['name', 'sibsp', 'parch', 'ticket', 'cabin', 'family_size', 'home.dest', 'boat', 'body']
    X_train = X_train.drop(features_drop, axis=1)
    X_test = X_test.drop(features_drop, axis=1)
    X_train = X_train.drop(['age_band', 'fare_band'], axis=1)

    # ----------------------------------------- Define training and testing set ----------------------------------------

    X_train = X_train.drop('survived', axis=1)
    X_test = X_test.drop('survived', axis=1)
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    # ------------------------------------------- Classification and Accuracy ------------------------------------------

    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    y_pred_log_reg = clf.predict(X_test)
    acc_log_reg = round(clf.score(X_test, y_test) * 100, 2)
    print('acc_log_reg: ' + str(acc_log_reg) + ' percent')

    clf = SVC()
    clf.fit(X_train, y_train)
    y_pred_svc = clf.predict(X_test)
    acc_svc = round(clf.score(X_test, y_test) * 100, 2)
    print('acc_svc: ' + str(acc_svc) + ' percent')

    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train, y_train)
    y_pred_knn = clf.predict(X_test)
    acc_knn = round(clf.score(X_test, y_test) * 100, 2)
    print('acc_knn: ' + str(acc_knn) + ' percent')

    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred_decision_tree = clf.predict(X_test)
    acc_decision_tree = round(clf.score(X_test, y_test) * 100, 2)
    print('acc_decision_tree: ' + str(acc_decision_tree) + ' percent')

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    y_pred_random_forest = clf.predict(X_test)
    acc_random_forest = round(clf.score(X_test, y_test) * 100, 2)
    print('acc_random_forest: ' + str(acc_random_forest) + ' percent')


if __name__ == '__main__':
    zad_final()
