---
# Метаданные находятся в metadata.yaml.
# Сборка:
# pandoc -s report.md --metadata-file=metadata.yaml -o report.pdf \
#   --pdf-engine=xelatex --lua-filter=include-files.lua
---

# Введение и научная постановка

```{.include shift-heading-level-by=1}
docs/SCIENTIFIC_SCOPE.md
```

# Глава 1. Обзор литературы и полевые аналоги

```{.include shift-heading-level-by=1}
docs/Обзор литературы.md
```

# Глава 2. Геологические модели

```{.include shift-heading-level-by=1}
models/geological_model.md
models/model_parameters.md
```

# Глава 3. Методы

```{.include shift-heading-level-by=1}
methods/dc_method.md
methods/ip_method.md
methods/petroleum_resistivity_logging.md
```

# Глава 4. Численное моделирование

```{.include shift-heading-level-by=1}
modeling/numerical_modeling.md
```

# Глава 5. Результаты и обсуждение

```{.include shift-heading-level-by=1}
docs/RESULTS.md
docs/FIELD_COMPARISON.md
```

![Типизированная золото-сульфидная модель](outputs/reference_model.png){ width=90% }

![Синтетический DC-псевдоразрез](outputs/dc_pseudosection.png){ width=90% }

![Синтетический IP-псевдоразрез](outputs/ip_pseudosection.png){ width=90% }

![Сходимость DC/IP на вложенных сетках](outputs/mesh_convergence.png){ width=80% }

![Ошибки оценки параметров двух моделей](outputs/dcip_parameter_recovery.png){ width=90% }

# Глава 6. Частный случай нефтегазовых зон Камеруна

```{.include shift-heading-level-by=1}
docs/PETROLEUM_GAS_CASE.md
```

![Оценка параметров нефтегазового резервуара Rio del Rey](outputs/petroleum_parameter_recovery.png){ width=90% }

# Заключение

Работа объединяет три согласованных направления: оценку параметров
золото-сульфидной зоны методом DC/IP, оценку канала железной минерализации теми
же методами и, как частный случай, исследование нефтегазового резервуара по
глубинному каротажу сопротивления. Для контролируемых синтетических данных
ширина, сопротивление и зарядоспособность восстановлены с медианными ошибками
около 6–10 %. Глубина кровли восстанавливается хуже: ошибки составляют 33 %
для золото-сульфидной и 50 % для железной модели при принятой поисковой сетке.

Все восстановленные значения согласуются с полевыми интервалами Bindiba и
Messondo. Следовательно, совместные DC/IP данные обладают практическим
потенциалом для оценки электрических свойств и горизонтального масштаба целей,
однако глубину следует ограничивать дополнительными геологическими данными или
более детальной инверсией. Главный результат — количественная оценка точности
параметров, а не бинарное заключение об обнаружимости цели.

Для резервуара Rio del Rey кровля оценена с абсолютной ошибкой 2 м, мощность —
с ошибкой 5,9 %, сопротивление — 2,9 %, а условная водонасыщенность — 1,8 %.
Сопротивление само по себе не различает нефть и газ, поэтому вывод ограничен
геометрией и петрофизическими параметрами резервуара.

# Литература

```{.include shift-heading-level-by=1}
docs/REFERENCES.md
```
