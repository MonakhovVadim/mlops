import os

from sklearn.preprocessing import MinMaxScaler
import pandas as pd


def main():

    current_file = os.path.realpath(__file__)
    current_directory = os.path.dirname(current_file)
    data_directory = os.path.split(current_directory)[0] + "/data"

    # Загружаем данные
    train_data = pd.read_csv(data_directory + "/train/train_data.csv")
    test_data = pd.read_csv(data_directory + "/test/test_data.csv")

    # Используем MinMaxScaler, чтобы не было отрицательных значений
    scaler = MinMaxScaler()
    train_data["value"] = scaler.fit_transform(train_data[["value"]])
    test_data["value"] = scaler.transform(test_data[["value"]])

    # Сохраняем предобработанные данные
    train_data.to_csv(
        data_directory + "/train/train_data_preprocessed.csv", index=False
    )
    test_data.to_csv(data_directory + "/test/test_data_preprocessed.csv", index=False)


if __name__ == "__main__":
    main()
