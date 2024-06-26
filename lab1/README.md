# Практическое задание №1

В ходе практической работы был создан простейший конвейер для автоматизации работы с моделью машинного обучения.
Отдельные этапы конвейера машинного обучения описываются в разных python–скриптах, которые потом соединяются с 
помощью bash-скрипта. В качестве примера используется модель, которая предсказывает курс доллара к рублю.

### Запуск скрипта
В папке scripts содержится pipeline.sh, который реализует конвейер машинного обучения. Он создает виртуальное
окружение и устанавливает необходимые библиотеки из файла requirements.txt, который находится в папке lab1.
Перед запуском скрипта следует убедиться, что в системе установлен python, а для debian-подобных систем, включая
ubuntu, установлен пакет python3-venv. В случае его отсутствия, следует воспользоваться командой 
`sudo apt-get install python3-venv` для его установки.

### Этапы скрипта
- data_creation.py генерирует случайные курсы валюты и сохраняет в папки data/test и data/train
- model_preprocessing.py предобрабатывает данные из предыдущего пункта и сохраняет в папки data/test и data/train
- model_preparation.py обучает модель на предобработанных данных и сохраняет модель в папку model
- model_testing.py - загружает сохраненную модель и тестирует ее с помощью метрик MSE и r2
