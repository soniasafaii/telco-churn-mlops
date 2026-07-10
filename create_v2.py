import pandas as pd

from src.preprocessing import preprocess


data = pd.read_excel(
    "data/v1/Telco_customer_churn.xlsx"
)


processed_data = preprocess(data)


processed_data.to_csv(
    "data/v2/Telco_customer_churn_v2.csv",
    index=False
)


print(processed_data.shape)
print(processed_data.head())
print("Churn Value" in processed_data.columns)
print(processed_data.isnull().sum().sum())
print(processed_data.columns)