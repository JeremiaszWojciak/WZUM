import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


# TODO 2 ---------------------------------------------------------------------------------------------------------------

def zad_2():
    digits = datasets.load_digits()
    # print(digits)
    print(f'DESCR:\n, {digits.DESCR}')
    print(f'data:\n {digits.data},\n len: {len(digits.data)}')
    print(f'target:\n {digits.target},\n len: {len(digits.target)}')
    print(f'target_names:\n {digits.target_names}')
    print(f'images:\n {digits.images},\n len: {len(digits.images)}')

    plt.imshow(digits.images[0], cmap='gray')
    plt.show()

    fig, axs = plt.subplots(len(digits.target_names), 5)
    for class_n in digits.target_names:
        for col in range(5):
            axs[class_n][col].imshow(digits.images[digits.target == class_n][col], cmap='gray')
            axs[class_n][col].axis('off')
    plt.show()


# TODO 3 ---------------------------------------------------------------------------------------------------------------

def zad_3():
    digits = datasets.load_digits()
    X, y = digits.data, digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(len(X))
    print(len(X_train))
    print(len(X_test))


# TODO 4 ---------------------------------------------------------------------------------------------------------------

def zad_4():
    faces = datasets.fetch_olivetti_faces()
    # alternative:
    #  X, y = datasets.fetch_olivetti_faces(return_X_y=True)

    # print(faces)

    X, y = faces.data, faces.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(y_test)

    image_shape = (64, 64)
    n_row, n_col = 2, 3
    n_components = n_row * n_col

    def plot_gallery(title, images, col=n_col, row=n_row, cmap='gray'):
        plt.figure(figsize=(2. * col, 2.26 * row))
        plt.suptitle(title, size=16)
        for i, comp in enumerate(images):
            plt.subplot(row, col, i + 1)
            vmax = max(comp.max(), -comp.min())
            plt.imshow(comp.reshape(image_shape), cmap=cmap,
                       interpolation='nearest',
                       vmin=-vmax, vmax=vmax)
            plt.xticks(())
            plt.yticks(())
        plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)

    plot_gallery('Olivetti faces', X_test[:n_components])
    plt.show()


# TODO 5 ---------------------------------------------------------------------------------------------------------------

def zad_5():
    irises = datasets.load_iris()
    # print(irises)

    sepal_l, sepal_w, petal_l, petal_w = [], [], [], []

    for class_n in range(len(irises.target_names)):
        class_data = irises.data[irises.target == class_n]
        sepal_l.append(class_data[:, 0])
        sepal_w.append(class_data[:, 1])
        petal_l.append(class_data[:, 2])
        petal_w.append(class_data[:, 3])

    fig, axs = plt.subplots(1, 4, figsize=(16, 7))
    axs[0].set_title('Sepal l')
    axs[1].set_title('Sepal w')
    axs[2].set_title('Petal l')
    axs[3].set_title('Petal w')

    bplot0 = axs[0].boxplot(sepal_l, labels=irises.target_names, positions=[1, 2, 3], patch_artist=True)
    bplot1 = axs[1].boxplot(sepal_w, labels=irises.target_names, positions=[1, 2, 3], patch_artist=True)
    bplot2 = axs[2].boxplot(petal_l, labels=irises.target_names, positions=[1, 2, 3], patch_artist=True)
    bplot3 = axs[3].boxplot(petal_w, labels=irises.target_names, positions=[1, 2, 3], patch_artist=True)

    for bplot in (bplot0, bplot1, bplot2, bplot3):
        bplot['boxes'][0].set_facecolor('pink')
        bplot['boxes'][1].set_facecolor('lightblue')
        bplot['boxes'][2].set_facecolor('lightgreen')
    plt.show()

    X, y = irises.data, irises.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(y_test)


# TODO 6 ---------------------------------------------------------------------------------------------------------------

def zad_6():
    m_class = datasets.make_classification(n_samples=50, n_features=2, n_informative=2, n_redundant=0, n_classes=2,
                                           n_clusters_per_class=1)
    print(m_class)

    X, y = m_class[0], m_class[1]
    plt.scatter(X[:, 0], X[:, 1], c=y, marker="o", edgecolor="k")
    plt.show()


# TODO 7 ---------------------------------------------------------------------------------------------------------------

def zad_7():
    credit = datasets.fetch_openml(data_id=31)
    print(credit)
    print(type(credit.data))
    print(type(credit.target))
    print(f'Sample data:\n {credit.data.iloc[0]}')
    print(f'Sample target:\n {credit.target.iloc[0]}')


# TODO 8 ---------------------------------------------------------------------------------------------------------------

def zad_8():
    df = pd.read_csv('trainingdata.txt', header=None)
    print(df)
    plt.scatter(df[0], df[1])
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(df[0], df[1])
    print(len(X_train), len(X_test))


# TODO Bramki logiczne -------------------------------------------------------------------------------------------------

def zad_bramki():
    X = [[0, 0],
         [0, 1],
         [1, 0],
         [1, 1]]
    y = [0, 0, 0, 1]

    clf = DecisionTreeClassifier()
    clf.fit(X, y)
    print(clf.predict([[1, 1]]))


# TODO 9 AND 10 --------------------------------------------------------------------------------------------------------

def zad_9_10():
    X = [[0, 0],
         [0, 1],
         [1, 0],
         [1, 1]]
    y = [0, 1, 1, 1]

    clf = DecisionTreeClassifier()
    clf = clf.fit(X, y)
    print(f'[0, 0]: {clf.predict([[0, 0]])}')
    print(f'[0, 1]: {clf.predict([[0, 1]])}')
    print(f'[1, 0]: {clf.predict([[1, 0]])}')
    print(f'[1, 1]: {clf.predict([[1, 1]])}')

    plot_tree(clf, feature_names=['X1', 'X2'], filled=True, class_names=['0', '1'])
    plt.show()


if __name__ == '__main__':
    zad_9_10()
