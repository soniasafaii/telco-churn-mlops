from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.model_selection import StratifiedKFold, cross_val_score


RANDOM_STATE = 42



def split_data(df):
    """
    Split data into train, validation and test sets.
    """

    X = df.drop(
        columns=["Churn Value"]
    )

    y = df["Churn Value"]


    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y
    )


    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )


    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test
    )



def get_models():
    """
    Return all machine learning models used in the project.
    """

    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=3000,
            random_state=RANDOM_STATE
        ),


        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE
        ),


        "XGBoost": XGBClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            eval_metric="logloss"
        ),


        "CatBoost": CatBoostClassifier(
            iterations=100,
            random_state=RANDOM_STATE,
            verbose=0
        )

    }


    return models



def cross_validate_models(models, X_train, y_train):
    """
    Perform Stratified K-Fold Cross Validation
    for all models.
    """


    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )


    results = {}


    for name, model in models.items():

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="f1"
        )


        results[name] = {
            "mean_f1": scores.mean(),
            "std_f1": scores.std()
        }


        print(
            f"{name}: "
            f"F1={scores.mean():.4f}"
        )


    return results



def train_model(model, X_train, y_train):
    """
    Train selected model on training data.
    """

    model.fit(
        X_train,
        y_train
    )

    return model

