# Описание проекта
Цель этого проекта — провести систематическое сравнительное исследование согласования с нелинейными ограничениями (Non-linearly Constrained Reconciliation, NLCR) — метода, который корректирует прогнозы для нескольких временных рядов так, чтобы они удовлетворяли заданным нелинейным ограничениям, а также смешанным линейным и нелинейным ограничениям, посредством решения задачи проекции/оптимизации. В экспериментах изучается, для каких типов рядов, включая зашумлённые, почти детерминированные, имеющие физический смысл и эконометрические ряды, и каких классов ограничений, включая соотношения, балансы, неравенства и границы, метод повышает точность, а в каких случаях ухудшает результаты. Для сравнения используются единые метрики и статистические тесты оценки прогнозов.

## Ссылка на статью
https://arxiv.org/pdf/2510.21249

## Реализованные методы NLCR
`src/nlcr.py` поддерживает три варианта ковариационной матрицы из статьи:

- `ols`: стандартная евклидова проекция, `W = I`;
- `wls`: диагональная ковариационная матрица ошибок базовых прогнозов;
- `shr`: усадочная оценка, объединяющая полную ковариационную матрицу ошибок с её диагональной WLS-версией.

Пример использования:

```python
from src.nlcr import reconcile_many

errors = y_train_true - y_train_base
y_reconciled_wls, failed_wls = reconcile_many(
    y_base,
    constraints,
    method="wls",
    errors=errors,
)
y_reconciled_shr, failed_shr = reconcile_many(
    y_base,
    constraints,
    method="shr",
    errors=errors,
)
```

## Локальная подготовка реальных наборов данных
В экспериментальных блокнотах сейчас используются следующие источники данных:

- `01_synthetic_noise.ipynb`: полностью синтетические данные, создаваемые внутри блокнота;
- `02_household_power.ipynb`: `data/processed/household_power/household_power.parquet`;
- `03_m5_sales.ipynb`: `data/processed/m5/m5_selected_hierarchy.parquet`;
- `04_lobster_bid_ask.ipynb`: `data/processed/lobster/lobster_bid_ask_1level.parquet`;
- `05_lobster_backtest.ipynb`: тот же подготовленный parquet-файл LOBSTER;
- `06_summary.ipynb`: таблицы результатов, созданные предыдущими экспериментальными блокнотами.

Исходные файлы хранятся в `data/raw/`, а готовые для блокнотов parquet-файлы —
в `data/processed/`. Как исходные наборы данных, так и parquet-файлы игнорируются
git и должны быть подготовлены локально.

Загрузите внешние исходные наборы данных:

```bash
python scripts/download_data.py household_power
python scripts/download_data.py m5 --source huggingface
python scripts/download_data.py lobster
```

Загрузчик M5 также может использовать Kaggle с помощью `--source kaggle` или
заданного по умолчанию `--source auto`; выше показано зеркало Hugging Face,
поскольку оно работает без учётных данных Kaggle.

Подготовьте parquet-файлы, используемые блокнотами:

```bash
python scripts/prepare_data.py household_power
python scripts/prepare_data.py m5 --n-items 3 --store-id CA_1
python scripts/prepare_data.py lobster --levels 1 --sample-every 20 --max-rows 2000
```

После подготовки блокноты `02_household_power.ipynb`, `03_m5_sales.ipynb` и
`04_lobster_bid_ask.ipynb` автоматически считывают реальные данные из
`data/processed/`; если соответствующий parquet-файл отсутствует, они используют
встроенные демонстрационные генераторы. `05_lobster_backtest.ipynb` намеренно
запускается только на подготовленной выборке LOBSTER и требует наличия файла
`data/processed/lobster/lobster_bid_ask_1level.parquet`.

# Автор
Пожидаев Филипп Александрович
