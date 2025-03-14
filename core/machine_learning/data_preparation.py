import random
import numpy as np
from sklearn.model_selection import train_test_split


def sample_data(data: tuple, fraction: float) -> tuple:
    if fraction != 1:
        x, y = data
        sample_size = int(len(x) * fraction)

        indices = random.sample(range(len(x)), sample_size)
        x_subset = np.array([x[i] for i in indices])
        y_subset = np.array([y[i] for i in indices])

        data = (x_subset, y_subset)

    return data


def split_data_by_proportions(data: tuple, parameters: dict) -> dict:
    x, y = data
    proportions = np.array(parameters["split_proportions"])

    if not validate_proportions(proportions):
        raise ValueError("Proportions must add up to 1.")
    
    valid_test_share = proportions[1:].sum()
    test_share = proportions[2] / valid_test_share

    X_train, X_temp, y_train, y_temp = train_test_split(
        x, y, test_size=valid_test_share, random_state=parameters["seed"]
    )

    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=test_share, random_state=parameters["seed"]
    )

    return {
        "train": (X_train, y_train),
        "valid": (X_valid, y_valid),
        "test" : (X_test, y_test)
    }


def validate_proportions(proportions):
    return np.round(proportions.sum(), 2) == 1