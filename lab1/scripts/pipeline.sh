#!/bin/bash

echo "Создание виртуального окружения и установка библиотек"
python3 -m venv ../../.venv
source ../../.venv/bin/activate
pip install -r ../requirements.txt

echo "Работа пайплайна..."

python ../src/data_creation.py
python ../src/model_preprocessing.py
python ../src/model_preparation.py
python ../src/model_testing.py

echo "Работа скрипта завершена"
