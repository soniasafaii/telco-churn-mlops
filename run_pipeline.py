def run_pipeline(dataset_version):

    print(f"Running pipeline for {dataset_version}")

    data = load_data(dataset_version)

    print("Loaded data:")
    print(data.shape)

    return data
from src.data_loader import load_data

