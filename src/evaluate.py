from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)



def evaluate_model(model, X, y):
    """
    Evaluate trained model.
    """


    y_pred = model.predict(X)


    # probability for ROC-AUC
    if hasattr(model, "predict_proba"):

        y_prob = model.predict_proba(X)[:,1]

    else:

        y_prob = None



    metrics = {

        "accuracy": accuracy_score(
            y,
            y_pred
        ),


        "precision": precision_score(
            y,
            y_pred
        ),


        "recall": recall_score(
            y,
            y_pred
        ),


        "f1_score": f1_score(
            y,
            y_pred
        )

    }



    if y_prob is not None:

        metrics["roc_auc"] = roc_auc_score(
            y,
            y_prob
        )


    cm = confusion_matrix(
        y,
        y_pred
    )


    return metrics, cm
def evaluate_on_validation(models, X_train, y_train, X_validation, y_validation):

    results = {}

    best_model = None
    best_model_name = None
    best_f1 = 0


    for name, model in models.items():

        print("=" * 40)
        print(name)


        model.fit(
            X_train,
            y_train
        )


        y_pred = model.predict(
            X_validation
        )


        metrics = {

            "accuracy": accuracy_score(
                y_validation,
                y_pred
            ),

            "precision": precision_score(
                y_validation,
                y_pred
            ),

            "recall": recall_score(
                y_validation,
                y_pred
            ),

            "f1_score": f1_score(
                y_validation,
                y_pred
            )
        }


        results[name] = metrics


        print(metrics)


        if metrics["f1_score"] > best_f1:

            best_f1 = metrics["f1_score"]

            best_model = model

            best_model_name = name


    return (
        best_model,
        best_model_name,
        results
    )
def evaluate_final_test(model, X_test, y_test):
    """
    Final evaluation on test data.
    This function should be called only once
    after selecting the best model.
    """

    y_pred = model.predict(
        X_test
    )


    if hasattr(model, "predict_proba"):

        y_prob = model.predict_proba(
            X_test
        )[:, 1]

    else:

        y_prob = None



    metrics = {

        "accuracy": accuracy_score(
            y_test,
            y_pred
        ),

        "precision": precision_score(
            y_test,
            y_pred
        ),

        "recall": recall_score(
            y_test,
            y_pred
        ),

        "f1_score": f1_score(
            y_test,
            y_pred
        )
    }



    if y_prob is not None:

        metrics["roc_auc"] = roc_auc_score(
            y_test,
            y_prob
        )



    cm = confusion_matrix(
        y_test,
        y_pred
    )


    return metrics, cm