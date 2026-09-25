"""
House Price Prediction using XGBoost Regressor
===============================================
This script predicts house prices using the Boston Housing Dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
import warnings

warnings.filterwarnings("ignore")


def load_boston_dataset():
    """Load Boston Housing Dataset with fallback for newer scikit-learn versions."""
    try:
        from sklearn.datasets import load_boston
        return load_boston()
    except ImportError:
        # Fallback for scikit-learn >= 1.2
        data_url = "http://lib.stat.cmu.edu/datasets/boston"
        raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
        data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
        target = raw_df.values[1::2, 2]
        feature_names = np.array(
            [
                "CRIM",
                "ZN",
                "INDUS",
                "CHAS",
                "NOX",
                "RM",
                "AGE",
                "DIS",
                "RAD",
                "TAX",
                "PTRATIO",
                "B",
                "LSTAT",
            ]
        )

        class Bunch:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        return Bunch(
            data=data,
            target=target,
            feature_names=feature_names,
            DESCR="Boston Housing Dataset",
        )


def main():
    print("=" * 50)
    print("House Price Prediction using XGBoost")
    print("=" * 50)

    # Load dataset
    print("\n[1/6] Loading dataset...")
    house_price_dataset = load_boston_dataset()
    print("Dataset loaded successfully!")

    # Create DataFrame
    print("\n[2/6] Creating DataFrame...")
    house_price_dataframe = pd.DataFrame(
        house_price_dataset.data, columns=house_price_dataset.feature_names
    )
    house_price_dataframe["price"] = house_price_dataset.target

    print(f"Dataset shape: {house_price_dataframe.shape}")
    print(f"\nFirst 5 rows:\n{house_price_dataframe.head()}")
    print(f"\nMissing values:\n{house_price_dataframe.isnull().sum()}")
    print(f"\nStatistical summary:\n{house_price_dataframe.describe()}")

    # Correlation heatmap
    print("\n[3/6] Generating correlation heatmap...")
    correlation = house_price_dataframe.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        correlation,
        cbar=True,
        square=True,
        fmt=".1f",
        annot=True,
        annot_kws={"size": 8},
        cmap="Blues",
    )
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png", dpi=150)
    print("Saved: correlation_heatmap.png")

    # Split data
    print("\n[4/6] Splitting data into train/test...")
    X = house_price_dataframe.drop(["price"], axis=1)
    Y = house_price_dataframe["price"]
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=2
    )
    print(f"Train set: {X_train.shape} | Test set: {X_test.shape}")

    # Train model
    print("\n[5/6] Training XGBoost Regressor...")
    model = XGBRegressor(objective="reg:squarederror", random_state=2)
    model.fit(X_train, Y_train)
    print("Model trained successfully!")

    # Evaluate
    print("\n[6/6] Evaluating model...")

    # Training metrics
    training_data_prediction = model.predict(X_train)
    train_r2 = metrics.r2_score(Y_train, training_data_prediction)
    train_mae = metrics.mean_absolute_error(Y_train, training_data_prediction)
    print(f"\n--- Training Results ---")
    print(f"R squared error : {train_r2:.4f}")
    print(f"Mean Absolute Error : {train_mae:.4f}")

    # Test metrics
    test_data_prediction = model.predict(X_test)
    test_r2 = metrics.r2_score(Y_test, test_data_prediction)
    test_mae = metrics.mean_absolute_error(Y_test, test_data_prediction)
    print(f"\n--- Test Results ---")
    print(f"R squared error : {test_r2:.4f}")
    print(f"Mean Absolute Error : {test_mae:.4f}")

    # Scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(Y_train, training_data_prediction, alpha=0.6)
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Actual Price vs Predicted Price (Training Data)")
    plt.plot(
        [Y_train.min(), Y_train.max()],
        [Y_train.min(), Y_train.max()],
        "r--",
        lw=2,
    )
    plt.tight_layout()
    plt.savefig("actual_vs_predicted.png", dpi=150)
    print("\nSaved: actual_vs_predicted.png")

    print("\n" + "=" * 50)
    print("Done! Check the generated plots.")
    print("=" * 50)


if __name__ == "__main__":
    main()
