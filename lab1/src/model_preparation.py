import os

import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib


def main():
    # Создаем папку модели
    os.makedirs("../data/model", exist_ok=True)

    # Загружаем данные
    data = pd.read_csv("../data/train/train_data_preprocessed.csv")
    X, y = data[["day"]], data["value"]

    # Обучаем модель
    model = LinearRegression()
    model.fit(X, y)

    # Сохраняем модель
    joblib.dump(model, "../data/model/model.pkl")


if __name__ == "__main__":
    main()
