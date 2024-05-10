import os

import numpy as np
import pandas as pd
import random


def generate_data(begin_val, count):
    """Генерирует данные, симулирующие курс доллара по отношению к рублю.
    Курс является случайным, но тяготеет к линии тренда тем сильнее, чем от
    нее отошел. В качестве линии тренда используются реальные данные за 20 лет.
    Также курс имеет склонность к продолжению роста/падения в зависимости от
    предыдущего значения.

    Args:
        begin_val (float): предначальное значение курса (до первой записи)
        count (int): количество записей

    Returns:
        DataFrame: сгенерированные данные с курсом валюты
    """

    data = pd.DataFrame()
    data["day"] = range(1, count + 1)

    # Задаем фиксированное зерно и получаем изменения курса из нормального
    # распределения.
    np.random.seed(33)
    incs = np.random.normal(scale=0.5, size=count)

    # Вычисляем налон линии тренда за 20 лет
    cur_begin = 29.0587
    cur_end = 91.6918
    interval = 20 * 365
    k_trend = (cur_end - cur_begin) / interval

    # Определяем в какую сторону будет движение курса
    directions = []
    value = begin_val
    prev_dir = 0
    for i in range(count):
        typ_val = begin_val + i * k_trend
        diff = 0 if typ_val == 0 else int((typ_val - value) * 50 / typ_val)

        score_sign = np.random.randint(low=-100, high=100) + prev_dir + diff
        sign_dir = 1 if score_sign >= 0 else -1
        directions.append(sign_dir)
        prev_dir = sign_dir * 80

    # Зная изменение и направление определяем значение курса на каждый день
    values = []
    changes = [a * b for a, b in zip(incs, directions)]
    value = begin_val
    for change in changes:
        value += change
        values.append(value)

    data["value"] = values

    return data


def main():

    current_file = os.path.realpath(__file__)
    current_directory = os.path.dirname(current_file)
    data_directory = os.path.split(current_directory)[0] + "/data"

    # Создаем папки
    os.makedirs("f{data_directory}/train", exist_ok=True)
    os.makedirs("f{data_directory}/test", exist_ok=True)

    # Генерируем курс валюты на ближайший год
    count = 365
    data = generate_data(91, count)

    # Получаем индексы для тестовых и тренировочных данных
    all_indexes = list(range(count))
    test_indexes = random.sample(all_indexes, int(count * 0.3))
    train_indexes = [x for x in all_indexes if not x in test_indexes]

    # Сохраняем данные в соответствующие папки
    data.iloc[test_indexes].to_csv("f{data_directory}/test/test_data.csv", index=False)
    data.iloc[train_indexes].to_csv(
        "f{data_directory}/train/train_data.csv", index=False
    )


if __name__ == "__main__":
    main()
