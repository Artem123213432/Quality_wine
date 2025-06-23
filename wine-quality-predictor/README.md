# Вин QA: Предсказание качества вина

Этот проект реализует систему прогнозирования качества вина на основе его химических свойств с использованием методов машинного обучения.

## Структура проекта

```
wine-quality-predictor/
├── data/                    # Директория с данными
│   └── winequality-red.csv  # Набор данных о красном вине
├── model/                   # Директория с моделями
│   └── train_model.py       # Скрипт для обучения модели
├── notebooks/               # Jupyter ноутбуки для исследования данных
│   └── exploratory_analysis.ipynb
├── results/                 # Директория для сохранения результатов
├── utils.py                 # Вспомогательные функции
├── predict.py               # Скрипт для предсказания качества вина
└── README.md                # Документация проекта
```

## Установка

```bash
# Клонирование репозитория
git clone <repository-url>
cd wine-quality-predictor

# Создание и активация виртуальной среды (опционально)
python -m venv venv
source venv/bin/activate  # Для Linux/Mac
# или
venv\Scripts\activate     # Для Windows

# Установка зависимостей
pip install -r requirements.txt
```

## Использование

### 1. Исследовательский анализ данных

Для запуска исследования данных в интерактивном режиме:

```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

### 2. Обучение модели

Чтобы обучить модель нейронной сети для предсказания качества вина:

```bash
cd wine-quality-predictor
python -m model.train_model
```

После выполнения скрипта в директории `results` будут сохранены:
- Обученная модель (`wine_quality_model.pth`)
- Нормализатор данных (`scaler.pkl`)
- Файл с метриками качества (`metrics.txt`)
- Графики обучения и оценки модели:
  - График обучения (`learning_curves.png`)
  - График предсказаний (`predictions.png`)
  - Гистограмма ошибок (`error_histogram.png`)
  - График остатков (`residuals.png`)

### 3. Предсказание качества вина

#### Предсказание для одного образца:

```bash
python predict.py --input "7.4,0.7,0.0,1.9,0.076,11,34,0.9978,3.51,0.56,9.4"
```

Где числа - это значения характеристик вина в следующем порядке:
- fixed acidity
- volatile acidity
- citric acid
- residual sugar
- chlorides
- free sulfur dioxide
- total sulfur dioxide
- density
- pH
- sulphates
- alcohol

#### Предсказание для нескольких образцов из CSV файла:

```bash
python predict.py --input data/new_samples.csv --output results/predictions.csv
```

CSV файл должен иметь те же заголовки столбцов, что и оригинальный набор данных.

## Настройка параметров предсказания

```bash
python predict.py --help
```

Доступные параметры:
- `--model`: путь к файлу модели (по умолчанию: 'results/wine_quality_model.pth')
- `--scaler`: путь к файлу нормализатора (по умолчанию: 'results/scaler.pkl')
- `--input`: путь к CSV файлу с данными или строка с параметрами через запятую
- `--output`: путь для сохранения результатов предсказания (только для CSV входных данных)

## Зависимости

- Python 3.6+
- PyTorch
- NumPy
- pandas
- Matplotlib
- scikit-learn
- Jupyter (для ноутбуков) 