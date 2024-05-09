from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import pandas as pd
import joblib


def main():
    # Загружаем модель и данные для тестов
    model = joblib.load("../data/model/model.pkl")
    test_data = pd.read_csv("../data/test/test_data_preprocessed.csv")
    X_test, y_test = test_data[["day"]], test_data["value"]

    # Оценка модели
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("Метрики при тестировании модели:")
    print(f"MSE: {mse}")
    print(f"r2: {r2}")
    print("Что-то метрики не впечатляют)")


if __name__ == "__main__":
    main()
