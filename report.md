---
# Метаданные в отдельном файле metadata.yaml
# Этот файл служит только для сборки частей вместе.
# 
# Сборка через pandoc:
# pandoc -s report.md --metadata-file=metadata.yaml -o report.pdf --pdf-engine=xelatex --lua-filter=include-files.lua
---


# Введение
```{.include shift-heading-level-by=1}
README.md
```

# Глава 1. Методы
```{.include shift-heading-level-by=1}
methods/tdem_method
methods/ip_method
methods/dc_method
```

# Глава 2. Моделирование

```{.include shift-heading-level-by=1}
modeling/numerical_modeling 
modeling/Numerical Modeling
```

# Глава 3. Результаты и обсуждение

# Заключение

# Литература
