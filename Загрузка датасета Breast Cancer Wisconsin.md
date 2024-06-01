parent: [[Освоение практического опыта оценки моделей и настройки гиперпараметров]]

tags: #ml #book_club #dataset 

```python
import pandas as pd

df = pd.read_csv('https://archive.ics.uci.edu/ml/'
                 'machine-learning-databases'
                 '/breast-cancer-wisconsin/wdbc.data', header=None)

df.head()
```

![[Screenshot 2023-12-23 at 15.02.33.png]]

Далее нужно трансформировать строковые представления меток классов в числовые с помощью `LabelEncoder`:

```python
from sklearn.preprocessing import LabelEncoder

X = df.loc[:, 2:].values
y = df.loc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)
le.classes_
```

```output
array(['B', 'M'], dtype=object)
```

```python
le.transform(['M', 'B'])
```

```output
array([1, 0])
```

Разделяем датасет на трейн и тест:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, 
                     test_size=0.20,
                     stratify=y,
                     random_state=1)
```

