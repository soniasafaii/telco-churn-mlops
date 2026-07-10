import pandas as pd
from sklearn.preprocessing import StandardScaler


def create_new_features(df):

    df["Average Monthly Spend"] = (
        df["Total Charges"] /
        (df["Tenure Months"] + 1)
    )

    service_columns = [
        "Phone Service_Yes",
        "Online Security_Yes",
        "Online Backup_Yes",
        "Device Protection_Yes",
        "Tech Support_Yes",
        "Streaming TV_Yes",
        "Streaming Movies_Yes"
    ]

    df["Total Services"] = df[service_columns].sum(axis=1)


    df["Is New Customer"] = (
        df["Tenure Months"] <= 12
    ).astype(int)


    df["High Monthly Charges"] = (
        df["Monthly Charges"] >
        df["Monthly Charges"].median()
    ).astype(int)


    df["Charge per Tenure"] = (
        df["Monthly Charges"] /
        (df["Tenure Months"] + 1)
    )


    return df



def normalize_features(df):

    numerical_columns = [
        "Tenure Months",
        "Monthly Charges",
        "Total Charges",
        "Average Monthly Spend",
        "Charge per Tenure"
    ]

    scaler = StandardScaler()

    df[numerical_columns] = scaler.fit_transform(
        df[numerical_columns]
    )

    return df



def feature_engineering(df):

    df = create_new_features(df)

    df = normalize_features(df)

    return df