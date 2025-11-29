import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
import yaml


def load_params(params_path: str = "params.yaml"):
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    return params


def main():
    # 1. Загружаем параметры
    params = load_params()
    prepare_params = params["prepare"]

    test_size = prepare_params["test_size"]
    random_state = prepare_params["random_state"]
    target_column = prepare_params["target_column"]

    # 2. Загружаем сырые данные
    raw_data_path = "data/raw/data.csv"
    df = pd.read_csv(raw_data_path)

    # 3. Делим на X и y
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # 4. train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # чтобы классы были более равномерно распределены
    )

    # 5. Собираем обратно в датафреймы (признаки + таргет)
    train_df = X_train.copy()
    train_df[target_column] = y_train

    test_df = X_test.copy()
    test_df[target_column] = y_test

    # 6. Создаём папку для обработанных данных
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 7. Сохраняем
    train_path = processed_dir / "train.csv"
    test_path = processed_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train shape: {train_df.shape}, saved to {train_path}")
    print(f"Test shape: {test_df.shape}, saved to {test_path}")


if __name__ == "__main__":
    main()
