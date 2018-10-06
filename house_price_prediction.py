from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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


def main(use_visualization=True):
    housing = load_housing_data()

    if use_visualization:
        histograms(housing)


if __name__ == "__main__":
    main()
