"""
Classifying Iris Dataset using KNeighboardClassifier
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

COLUMNS = ["SPL", "SPW", "PLT", "PTW", "target"]


def iris_dataframe(data):
    # we nee DataFrame only for plots all ML logic we can do it without it
    df = pd.DataFrame(data["data"])
    # add target column
    df = df.assign(target=data["target"])
    # rename columns, since the one in the dataset are too long
    df.columns = COLUMNS
    return df


def pairplot(df):
    # create pairplot for data nalysis
    # add all columns to the graph except of target column
    sns.pairplot(df, hue="target", vars=COLUMNS[:-1])
    plt.show()


def boxplot(df):
    g = sns.boxplot(data=df)
    g.figure.set_size_inches(16, 8)
    plt.show()


def model(X_train, y_train):

    clf = KNeighborsClassifier(n_neighbors=1)
    # build model.
    # fit method returns knn object itself (and modifies it in place)
    clf.fit(X_train, y_train)
    return clf


def test_model(clf, X_test, y_test):
    print(f"Test set score: {clf.score(X_test, y_test):.2f}")


def predictor(clf, data):
    """Predicts iris type"""

    prediction = clf.predict(data)
    print(f"Prediction: {prediction}")
    return prediction


def run(clf, target_names):
    X_new = np.array([[5, 2.9, 1, 0.2]])
    prediction = predictor(clf, X_new)
    print(f"Predicted target name: {target_names[prediction]}")


def main(use_visualization=True):
    # get iris data
    iris_dataset = load_iris()
    target_names = iris_dataset["target_names"]

    if use_visualization:
        iris_df = iris_dataframe(iris_dataset)

        # visualize data in pairplot
        pairplot(iris_df)
        # visualize data in boxplot
        boxplot(iris_df)

    # In scikit-learn, data is usually denoted with a capital X,
    # while labels are denoted by a lowercase y.
    # This is inspired by the standard formulation f(x)=y in mathematics,
    # where x is the input to a function and y is the output.
    # Following more conventions from mathematics,
    # we use a capital X because the data is a two-dimensional array (a matrix)
    # and a lowercase y because the target is a one-dimensional array (a vector).
    X_train, X_test, y_train, y_test = train_test_split(
        iris_dataset["data"], iris_dataset["target"], random_state=0
    )

    # train model and get it back
    clf = model(X_test, y_test)

    # test model
    test_model(clf, X_train, y_train)

    # run our test data
    run(clf, target_names)


if __name__ == "__main__":
    main()
