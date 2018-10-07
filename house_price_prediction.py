from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit


HOUSING_PATH = r"./datasets/housing.csv"
COLUMNS = [
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
    "median_house_value",
]


def load_housing_data(housing_path=HOUSING_PATH):
    """Loads data from file to DataFrame"""
    csv_path = Path(housing_path)
    return pd.read_csv(csv_path)


def histograms(df):
    rows, cols = (2, 4)
    _, axes = plt.subplots(rows, cols, figsize=(18, 12))
    for i, col in enumerate(COLUMNS):
        # turn kde off so it shows count instead of dencity
        # sns.distplot(df[col].dropna(), kde=False, bins=50, ax=axes[divmod(i, cols)])
        sns.distplot(df[col].dropna(), bins=50, ax=axes[divmod(i, cols)])

    plt.show()


def split_train_test(dataset):
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    train_index, test_index = next(split.split(dataset, dataset["income_cat"]))

    strat_train_set, strat_test_set = dataset.loc[train_index], dataset.loc[test_index]

    # remove the income_cat attribute so the data is back to its original state
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    return strat_train_set, strat_test_set


def income_cat_column(dataset):
    """
    We need to ensure that the test set is representative of the various categories of incomes in the whole dataset.
    Since the median income is a continuous numerical attribute,
    we first need to create an income category attribute.
    Most median income values are clustered around $20,000–$50,000,
    but some median incomes go far beyond $60,000.
    It is important to have a sufficient number of instances in your dataset for each stratum,
    or else the estimate of the stratum’s importance may be biased.
    This means that you should not have too many strata,
    and each stratum should be large enough.
    The following code creates an income category attribute by dividing the median income by 1.5
    (to limit the number of income categories), and rounding up using ceil (to have discrete categories),
    and then merging all the categories greater than 5 into category 5
    """

    # Divide by 1.5 to limit the number of income categories
    dataset["income_cat"] = np.ceil(dataset["median_income"] / 1.5)
    # Label those above 5 as 5
    dataset["income_cat"].where(dataset["income_cat"] < 5, 5.0, inplace=True)


def main(use_visualization=True):
    housing = load_housing_data()

    if use_visualization:
        histograms(housing)

    income_cat_column(housing)
    strat_train_set, strat_test_set = split_train_test(housing)

    breakpoint()


if __name__ == "__main__":
    main(False)
