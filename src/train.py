from pathlib import Path
import joblib
import yaml
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow


def load_params(params_path: str = "params.yaml"):
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    return params


def get_model(train_params: dict):
    model_type = train_params["model_type"]
    random_state = train_params["random_state"]

    if model_type == "logreg":
        logreg_params = train_params["logreg"]
        model = LogisticRegression(
            C=logreg_params["C"],
            max_iter=logreg_params["max_iter"],
            random_state=random_state,
            solver="lbfgs",
            multi_class="auto",
        )
    elif model_type == "rf":
        rf_params = train_params["rf"]
        model = RandomForestClassifier(
            n_estimators=rf_params["n_estimators"],
            max_depth=rf_params["max_depth"],
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return model, model_type


def main():
    # 1. Параметры
    params = load_params()
    prepare_params = params["prepare"]
    train_params = params["train"]

    target_column = prepare_params["target_column"]
    test_size = prepare_params["test_size"]

    # 2. Загружаем данные
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]

    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    # 3. Модель
    model, model_type = get_model(train_params)

    # 4. Обучение
    model.fit(X_train, y_train)

    # 5. Оценка
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")

    # 6. Сохраняем модель
    model_path = Path("model.pkl")
    joblib.dump(model, model_path)

    # 7. Логируем в MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("iris_hw1")

    with mlflow.start_run():
        # Параметры пайплайна
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", train_params["random_state"])

        # Гиперпараметры модели
        if model_type == "logreg":
            for k, v in train_params["logreg"].items():
                mlflow.log_param(f"logreg_{k}", v)
        elif model_type == "rf":
            for k, v in train_params["rf"].items():
                mlflow.log_param(f"rf_{k}", v)

        # Метрики
        mlflow.log_metric("accuracy", acc)

        # Артефакты
        mlflow.log_artifact(str(model_path))
        # Можно дополнительно:
        # mlflow.sklearn.log_model(model, artifact_path="model")

    print(f"Model saved to {model_path} and logged to MLflow.")


if __name__ == "__main__":
    main()
