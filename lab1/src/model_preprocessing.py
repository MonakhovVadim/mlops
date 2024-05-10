from sklearn.preprocessing import MinMaxScaler
import pandas as pd


def main():
    # Загружаем данные
    train_data = pd.read_csv("../data/train/train_data.csv")
    test_data = pd.read_csv("../data/test/test_data.csv")

    # Используем MinMaxScaler, чтобы не было отрицательных значений
    scaler = MinMaxScaler()
    scaler.fit_transform(pd.concat([train_data[["value"]], test_data[["value"]]]))

    train_data["value"] = scaler.transform(train_data[["value"]])
    test_data["value"] = scaler.transform(test_data[["value"]])

    # Сохраняем предобработанные данные
    train_data.to_csv("../data/train/train_data_preprocessed.csv", index=False)
    test_data.to_csv("../data/test/test_data_preprocessed.csv", index=False)


if __name__ == "__main__":
    main()
