# Каталог данных

Исходные и обработанные наборы данных не добавляются в git.

Ожидаемая структура:

```text
data/
  raw/
    synthetic/
    household_power/
    m5/
    lobster/
  processed/
    synthetic/
    household_power/
    m5/
    lobster/
  samples/
```

Для локальных экспериментов данные можно загрузить в пользовательский кеш,
например `~/.cache/nlcr-data/`, а затем загружать непосредственно в объекты DataFrame.

## Процесс локальной подготовки

Загрузите исходные файлы:

```bash
python scripts/download_data.py household_power
python scripts/download_data.py lobster
python scripts/download_data.py m5
```

`m5` использует интерфейс командной строки Kaggle и требует наличия
`~/.kaggle/kaggle.json`, а также принятия правил соревнования
`m5-forecasting-accuracy`. Если интерфейс командной строки Kaggle недоступен,
вручную поместите следующие файлы в `data/raw/m5/`:

```text
calendar.csv
sales_train_evaluation.csv
sell_prices.csv
```

Подготовьте parquet-файлы для блокнотов:

```bash
python scripts/prepare_data.py household_power
python scripts/prepare_data.py m5 --n-items 3 --store-id CA_1
python scripts/prepare_data.py lobster --levels 1 --sample-every 20 --max-rows 2000
```

Экспериментальные блокноты сначала ищут файлы в `data/processed/...`; если
обработанный файл отсутствует, они используют встроенный демонстрационный
генератор, чтобы блокнот можно было запустить на чистой машине.
