import pandas as pd
import argparse
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlflow.models.signature import infer_signature


def main(data_path, n_estimators, max_depth):
    # Load dataset
    df = pd.read_csv(data_path)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop("Churn", axis=1),
        df["Churn"],
        test_size=0.2,
        random_state=42
    )

    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Logging parameter dan metrik
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", acc)

        # signature dan contoh input
        input_example = X_train.iloc[:5]
        signature = infer_signature(X_train, model.predict(X_train))

    # Logging model dengan signature dan input example
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=input_example
        )

        print(f"Accuracy: {acc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str,
                        default="./Telco_preprocessed.csv")
    parser.add_argument("--n_estimators", type=int, default=257)
    parser.add_argument("--max_depth", type=int, default=25)
    args = parser.parse_args()

    main(args.data_path, args.n_estimators, args.max_depth)
