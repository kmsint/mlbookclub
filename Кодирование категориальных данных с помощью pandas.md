parent: [[Обработка категориальных данных]]

tags: #ml #mlbookclub #reading 

Создадим простой датайфрейм:

```python
import pandas as pd

df= pd.DataFrame([
    ['green', 'М', 10.1, 'class2'], 
    ['red', 'L', 13.5, 'class1'], 
    ['bluе', 'XL', 15.3, 'class2']])
df.columns = ['color', 'size', 'price', 'classlabel']
```

![[Screenshot 2023-12-08 at 17.02.11.png]]

объект DataFrame содержит столбцы с именным признаком (color), порядковым признаком (size) и числовым признаком (price).

1. [[Маппинг порядковых признаков]]
2. [[Кодирование меток классов]]
3. [[One-hot encoding на именных признаках]]
4. [[Кодирование порядковых признаков]]










