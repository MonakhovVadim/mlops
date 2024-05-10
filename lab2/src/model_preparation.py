import os

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib


def main():

    current_file = os.path.realpath(__file__)
    current_directory = os.path.dirname(current_file)
    data_directory = os.path.split(current_directory)[0] + "/data"

    # Создаем папку модели
    os.makedirs(data_directory + "/model", exist_ok=True)

    # Загружаем данные
    data = pd.read_csv(data_directory + "/train/train_data_preprocessed.csv")
    X, y = data[["day"]], data["value"]

    # Обучаем модель

    model = RandomForestRegressor(max_depth=2, random_state=0)
    model.fit(X, y)

    # Сохраняем модель
    joblib.dump(model, data_directory + "/model/model.pkl")


if __name__ == "__main__":
    main()
