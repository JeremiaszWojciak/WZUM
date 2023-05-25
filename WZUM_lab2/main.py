import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OrdinalEncoder, PolynomialFeatures
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, r2_score, \
    mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn import pipeline


# TODO 1 ---------------------------------------------------------------------------------------------------------------

def zad_1():
    digits = datasets.load_digits()
    X, y = digits.data, digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # clf = DecisionTreeClassifier()
    clf = SVC()
    clf.fit(X_train, y_train)
    print(clf.predict(X_test))
    print(clf.score(X_test, y_test))


# TODO 2 ---------------------------------------------------------------------------------------------------------------

def zad_2_1():
    irises = datasets.load_iris()

    X, y = irises.data, irises.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # clf = DecisionTreeClassifier()
    clf = SVC()
    # clf = KNeighborsClassifier()
    clf.fit(X_train, y_train)
    print(clf.predict(X_test))
    print(clf.score(X_test, y_test))


def zad_2_2():
    m_class = datasets.make_classification(n_samples=50, n_features=2, n_informative=2, n_redundant=0, n_classes=2,
                                           n_clusters_per_class=1)

    X, y = m_class[0], m_class[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # clf = DecisionTreeClassifier()
    clf = SVC()
    # clf = KNeighborsClassifier()
    clf.fit(X_train, y_train)
    print(clf.predict(X_test))
    print(clf.score(X_test, y_test))


# TODO 3 ---------------------------------------------------------------------------------------------------------------

def zad_3():
    mydataset = pd.read_csv('mydataset.txt')

    enc = OrdinalEncoder()
    mydataset[['Brand', 'Broken', 'Worth_buying']] = enc.fit_transform(mydataset[['Brand', 'Broken', 'Worth_buying']])

    X, y = mydataset.iloc[:, 0:3], mydataset.iloc[:, 3]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(type(X_train))
    print(type(y_train))

    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    print(clf.predict(X_test))
    print(clf.score(X_test, y_test))

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X[['Brand']].values, X[['Mileage']].values, X[['Broken']].values, c=y, marker="o", edgecolor="k")
    ax.set_xlabel('Brand')
    ax.set_ylabel('Mileage')
    ax.set_zlabel('Broken')
    plt.show()


# TODO 4 ---------------------------------------------------------------------------------------------------------------

def zad_4():
    digits = datasets.load_digits()

    X, y = digits.data, digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    print('Accuracy:', clf.score(X_test, y_test))

    cm = confusion_matrix(y_test, predicted)
    print('Confusion matrix:\n', cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    print('Classification report:\n', classification_report(y_test, predicted))

    wrong_3_bool = np.logical_and(y_test == 3, predicted != 3)
    wrong_8_bool = np.logical_and(y_test == 8, predicted != 8)
    wrong_3_img = X_test[wrong_3_bool]
    wrong_8_img = X_test[wrong_8_bool]
    min_dim = np.min([len(wrong_3_img), len(wrong_8_img)])

    fig, axs = plt.subplots(2, min_dim)
    for row in range(2):
        for col in range(min_dim):
            if row == 0:
                axs[row][col].imshow(np.reshape(wrong_3_img[col], (8, 8)), cmap='gray_r')
                axs[row][col].axis('off')
            elif row == 1:
                axs[row][col].imshow(np.reshape(wrong_8_img[col], (8, 8)), cmap='gray_r')
                axs[row][col].axis('off')
    plt.show()


# TODO 5 ---------------------------------------------------------------------------------------------------------------

def zad_5():
    df = pd.read_csv('trainingdata.txt', header=None)
    # plt.scatter(df[0], df[1])
    # plt.show()

    X, y = df[0].values.reshape(-1, 1), df[1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    reg = LinearRegression()
    reg.fit(X_train, y_train)
    predicted = reg.predict(X_test)
    print('Regression score: ', reg.score(X_test, y_test))

    plt.scatter(X_test, y_test, c='red', marker='o')
    plt.scatter(X_test, predicted, c='green', marker='*')
    plt.plot(X_test, predicted, c='green')
    plt.show()


# TODO 6 ---------------------------------------------------------------------------------------------------------------

def zad_6():
    df = pd.read_csv('trainingdata.txt', header=None)

    X, y = df[0].values.reshape(-1, 1), df[1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    lin_predicted = lin_reg.predict(X_test)

    print('Linear regressor:')
    print('MAE: ', mean_absolute_error(y_test, lin_predicted))
    print('MSE: ', mean_squared_error(y_test, lin_predicted))
    print('R2: ', r2_score(y_test, lin_predicted))

    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(X_train, y_train)
    tree_predicted = tree_reg.predict(X_test)

    print('Decision tree regressor:')
    print('MAE: ', mean_absolute_error(y_test, tree_predicted))
    print('MSE: ', mean_squared_error(y_test, tree_predicted))
    print('R2: ', r2_score(y_test, tree_predicted))

    poly_reg = pipeline.Pipeline(
        [
            ('poly', PolynomialFeatures(degree=9)),
            ('linear', LinearRegression())
        ]
    )
    poly_reg.fit(X_train, y_train)
    poly_predicted = poly_reg.predict(X_test)

    print('Polynomial regressor:')
    print('MAE: ', mean_absolute_error(y_test, poly_predicted))
    print('MSE: ', mean_squared_error(y_test, poly_predicted))
    print('R2: ', r2_score(y_test, poly_predicted))

    plt.scatter(X_test, y_test, c='red', marker='o')
    plt.scatter(X_test, lin_predicted, c='green', marker='*')
    plt.scatter(X_test, tree_predicted, c='orange', marker='+')
    plt.scatter(X_test, poly_predicted, c='blue', marker='.')
    plt.show()


if __name__ == '__main__':
    zad_6()
