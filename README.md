# Домашнее задание 1. Создание воспроизводимого ML-pipeline с использованием DVC и MLflow 
**Автор:** *Лаврухина Виктория*

---

## Цель проекта

Цель проекта — построить минимальный, но полноценный MLOps-контур, обеспечивающий:

- воспроизводимость экспериментов;
- управление версиями данных с помощью **DVC**;
- автоматизацию обучения модели через **DVC pipeline**;
- логирование параметров, метрик и артефактов в **MLflow**;
- запуск всего процесса одной командой:

```bash
dvc repro
```

## Структура проекта
```
├── data/                  # данные (хранятся через DVC)
│   ├── raw/               # сырые данные
│   └── processed/         # подготовленные данные
│
├── src/
│   ├── prepare.py         # подготовка и разбиение данных
│   └── train.py           # обучение модели и логирование MLflow
│
├── .dvc_storage/          # in-repo DVC remote (для воспроизводимости)
│
├── dvc.yaml               # DVC-пайплайн: prepare → train
├── params.yaml            # параметры обучения и препроцессинга
├── requirements.txt       # зависимости
├── README.md              # документация
```

## Как запустить проект
1. Склонируйте репозиторий:
```bash
git clone https://github.com/LavrukhinaV/MLOps_HW1.git
cd MLOps_HW1
```
2. Создайте и активируйте виртуальное окружение:
```bash
python3 -m venv venv
source venv/bin/activate       # macOS / Linux
# .\venv\Scripts\activate      # Windows
```
3. Установите зависимости:
```bash
pip install -r requirements.txt
```
4. Подтяните данные и артефакты из DVC:
```bash
dvc pull
```
5. Запустите ML-пайплайн::
```bash
dvc repro
```
После выполнения будет создана новая модель model.pkl и выполнено логирование в MLflow.
1. Запустить MLflow UI:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
2. Открыть в браузере:
```bash
http://127.0.0.1:5000
```

## Описание DVC-пайплайна
Пайплайн содержит две стадии:

1. **prepare**

Скрипт `src/prepare.py`:
* читает data/raw/data.csv;
* выполняет train/test split;
* сохраняет:
  * data/processed/train.csv
  * data/processed/test.csv.

2. **train**

Скрипт `src/train.py`:
* обучает модель (LogReg или RF);
* считает accuracy;
* сохраняет model.pkl;
* логирует параметры, метрики и артефакты в MLflow.

## Используемые технологии

**DVC** — контроль версий данных, управление пайплайнами

**MLflow** — эксперимент-трекинг

**scikit-learn** — модели машинного обучения

**pandas, numpy** — обработка данных

**Python 3**