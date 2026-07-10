import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def remove_unnecessary_columns(df):
    """
    Remove columns that are not useful for churn prediction.
    """

    columns_to_drop = [
        "CustomerID",
        "Count",
        "Country",
        "State",
        "City",
        "Zip Code",
        "Lat Long",
        "Latitude",
        "Longitude",
        "Churn Label",
        "Churn Score",
        "Churn Reason",
        "CLTV"
    ]

    df = df.drop(columns=columns_to_drop)

    return df


def convert_data_types(df):
    """
    Convert columns to proper data types.
    """

    df["Total Charges"] = pd.to_numeric(
        df["Total Charges"],
        errors="coerce"
    )

    return df


def handle_missing_values(df):
    """
    Handle missing values after type conversion.
    """

    df["Total Charges"] = df["Total Charges"].fillna(
        df["Total Charges"].median()
    )

    return df


def encode_features(df):
    """
    Encode categorical features.
    """

    categorical_columns = df.select_dtypes(
        include=["object"]
    ).columns

   
    df = pd.get_dummies(
    df,
    columns=categorical_columns,
    drop_first=True,
    dtype=int
    )

    return df


def preprocess(df):
    """
    Complete preprocessing pipeline.
    """

    df = remove_unnecessary_columns(df)

    df = convert_data_types(df)

    df = handle_missing_values(df)

    df = encode_features(df)

    return df

def prepare_raw_data_for_training(df):
    """
    Minimal preprocessing for raw dataset (v1)
    before sending data to machine learning models.
    """

    df = df.copy()


    # Columns that should not be used for training
    columns_to_drop = [
        "CustomerID",
        "Count",
        "City",
        "Zip Code",
        "Lat Long",
        "Latitude",
        "Longitude",
        "Churn Label",
        "Churn Score",
        "Churn Reason",
        "CLTV"
    ]


    df = df.drop(
        columns=columns_to_drop,
        errors="ignore"
    )


    # Convert Total Charges to numeric
    df["Total Charges"] = pd.to_numeric(
        df["Total Charges"],
        errors="coerce"
    )


    # Handle missing values
    df["Total Charges"] = df["Total Charges"].fillna(
        df["Total Charges"].median()
    )


    # Encode categorical columns
    categorical_columns = df.select_dtypes(
        include=["object"]
    ).columns


    df = pd.get_dummies(
        df,
        columns=categorical_columns,
        drop_first=True,
        dtype=int
    )


    return df