import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import ConfusionMatrixDisplay



def log_experiment(
        dataset_version,
        model_name,
        model,
        metrics,
        confusion_matrix
):

    mlflow.set_experiment(
        "Telco_Churn_Experiment"
    )


    with mlflow.start_run(
        run_name=f"{dataset_version}_{model_name}"
    ):


        # -----------------------
        # Parameters
        # -----------------------

        mlflow.log_param(
            "dataset_version",
            dataset_version
        )


        mlflow.log_param(
            "model_name",
            model_name
        )


        # random seed
        if hasattr(model, "random_state"):

            mlflow.log_param(
                "random_state",
                model.random_state
            )


        # model hyperparameters

        params = model.get_params()

        for key, value in params.items():

            try:

                mlflow.log_param(
                    key,
                    value
                )

            except:

                pass



        # -----------------------
        # Metrics
        # -----------------------

        for key, value in metrics.items():

            mlflow.log_metric(
                key,
                value
            )


        # -----------------------
        # Confusion Matrix
        # -----------------------

        fig, ax = plt.subplots()


        ConfusionMatrixDisplay(
            confusion_matrix
        ).plot(
            ax=ax
        )


        plt.title(
            "Confusion Matrix"
        )


        mlflow.log_figure(
            fig,
            "confusion_matrix.png"
        )


        plt.close()



        # -----------------------
        # Model Artifact
        # -----------------------

        mlflow.sklearn.log_model(
            model,
            name="model"
        )