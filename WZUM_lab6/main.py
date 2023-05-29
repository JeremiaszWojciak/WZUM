import mlflow
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
from mlflow import MlflowClient
from mlflow.entities import ViewType


# TODO 2 ---------------------------------------------------------------------------------------------------------------

def zad_2_1():
    mlflow.autolog()

    db = datasets.load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

    # Create and train models.
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
    rf.fit(X_train, y_train)

    # Use the model to make predictions on the test dataset.
    predictions = rf.predict(X_test)
    print(predictions)


def zad_2_2():
    db = datasets.load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

    model = mlflow.sklearn.load_model("mlruns/0/1192efbe7d354592a684bfb50f1350c3/artifacts/model")
    predictions = model.predict(X_test)
    print(predictions)


# TODO 3 AND 4 ---------------------------------------------------------------------------------------------------------

def zad_3_4():
    mlflow.set_experiment('my_experiment')
    mlflow.autolog()

    with mlflow.start_run():
        irises = datasets.load_iris()
        X, y = irises.data, irises.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X_train, y_train)

        predictions = clf.predict(X_test)
        print(predictions)

        mse = mean_squared_error(y_test, predictions)

        mlflow.log_metric('mean_squared_error', mse)
        mlflow.log_param('year', 2023)


# TODO 5 ---------------------------------------------------------------------------------------------------------------

# attributes.status = "FINISHED" and metrics.mean_squared_error > 0.060

def zad_5():
    run = MlflowClient().search_runs(
        experiment_ids=["707270387100777013"],
        order_by=["metrics.mean_squared_error DESC"],
        filter_string="metrics.mean_squared_error > 0.02"
    )
    print(run)


# TODO 6 ---------------------------------------------------------------------------------------------------------------

# mlflow experiments delete --experiment-id 707270387100777013
# mlflow gc --experiment-ids 707270387100777013


if __name__ == '__main__':
    zad_5()

