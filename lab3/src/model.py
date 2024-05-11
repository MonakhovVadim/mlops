from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def get_model():
    # Загружаем датасет
    dataset = load_iris()

    # Получаем имена предикторов и классов
    features = dataset.feature_names
    class_names = dataset.target_names

    # Деление на тренировочную и тустовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.2
    )

    # Создаем пайплайн с предобработкой и моделью
    model = Pipeline(
        [("scaler", StandardScaler()), ("RandomForest", RandomForestClassifier())]
    )

    # Обучаем модель
    model.fit(X_train, y_train)

    return model, features, class_names
