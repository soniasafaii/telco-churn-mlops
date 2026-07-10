import pandas as pd
import os


def load_data(version):
    """
    Load dataset based on version.
    
    version:
        v1 -> raw excel
        v2 -> preprocessed csv
        v3 -> feature engineered csv
    """

    if version == "v1":

        path = "data/v1/Telco_customer_churn.xlsx"

        df = pd.read_excel(path)


    elif version == "v2":

        path = "data/v2/Telco_customer_churn_v2.csv"

        df = pd.read_csv(path)


    elif version == "v3":

        path = "data/v3/Telco_customer_churn_v3.csv"

        df = pd.read_csv(path)


    else:
        raise ValueError(
            "Dataset version must be v1, v2 or v3"
        )


    return df