from src.data_loader import load_data

from src.preprocessing import (
    preprocess,
    prepare_raw_data_for_training
)

from src.features import feature_engineering

from src.train import (
    split_data,
    get_models,
    cross_validate_models,
    train_model
)

from src.evaluate import (
    evaluate_on_validation,
    evaluate_final_test
)



def run_pipeline(dataset_version):

    print("=" * 50)
    print(f"Running pipeline for {dataset_version}")
    print("=" * 50)


    # -------------------------
    # 1. Load Data
    # -------------------------

    df = load_data(dataset_version)

    print("Initial shape:")
    print(df.shape)



    # -------------------------
    # 2. Data Preparation
    # -------------------------

    if dataset_version == "v1":

        print("\nPreparing raw dataset...")

        df = prepare_raw_data_for_training(df)


    elif dataset_version == "v2":

        print("\nUsing preprocessed dataset...")

        df = df.copy()


    elif dataset_version == "v3":

        print("\nUsing feature engineered dataset...")

        df = df.copy()


    else:

        raise ValueError(
            "Dataset version must be v1, v2 or v3"
        )



    print("\nAfter preparation:")
    print(df.shape)



    # -------------------------
    # 3. Split Data
    # -------------------------

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)


    print("\nTrain:")
    print(X_train.shape)

    print("Validation:")
    print(X_val.shape)

    print("Test:")
    print(X_test.shape)



    # -------------------------
    # 4. Models
    # -------------------------

    models = get_models()



    # -------------------------
    # 5. Cross Validation
    # -------------------------

    print("\nCross Validation")

    cv_results = cross_validate_models(
        models,
        X_train,
        y_train
    )



    # -------------------------
    # 6. Model Selection
    # -------------------------

    print("\nModel Selection")


    best_model, best_name, validation_results = evaluate_on_validation(
        models,
        X_train,
        y_train,
        X_val,
        y_val
    )


    print("\nBest model:")
    print(best_name)



    # -------------------------
    # 7. Final Training
    # -------------------------

    best_model = train_model(
        best_model,
        X_train,
        y_train
    )



    # -------------------------
    # 8. Final Test
    # -------------------------

    print("\nFinal Test Evaluation")


    test_metrics, cm = evaluate_final_test(
        best_model,
        X_test,
        y_test
    )


    print(test_metrics)

    print("\nConfusion Matrix:")
    print(cm)



    return {
    "dataset_version": dataset_version,
    "best_model": best_name,
    "cv_results": cv_results,
    "validation_results": validation_results,
    "test_metrics": test_metrics,
    "confusion_matrix": cm
    }





if __name__ == "__main__":


    results_v1 = run_pipeline("v1")

    results_v2 = run_pipeline("v2")

    results_v3 = run_pipeline("v3")

