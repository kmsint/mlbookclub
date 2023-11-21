parent: [[Обучение простых ML-алгоритмов для классификации]]

tags: #reading #mlbookclub #ml 

```python
class Perceptron:
    """Perceptron classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.

    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    b_ : Scalar
      Bias unit after fitting.
    errors_ : list
      Number of misclassifications (updates) in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        Returns
        -------
        self : object

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.)
        
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, 0)
```

Рассмотрим код детально:

```python
def __init__(self, eta=0.01, n_iter=50, random_state=1):
	self.eta = eta
	self.n_iter = n_iter
	self.random_state = random_state
```

Здесь понятно - инициализация экземпляра класса. `eta` - это learning rate (от 0 до 1), `n_iter` - количество эпох (проходов по трейн-датасету), `random_state` - начальное значение генератора случайных чисел для инициализации персептрона случайными весами.

```python
def fit(self, X, y):
	rgen = np.random.RandomState(self.random_state)
	self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
	self.b_ = np.float_(0.)
	
	self.errors_ = []

	for _ in range(self.n_iter):
		errors = 0
		for xi, target in zip(X, y):
			update = self.eta * (target - self.predict(xi))
			self.w_ += update * xi
			self.b_ += update
			errors += int(update != 0.0)
		self.errors_.append(errors)
	return self
```

Метод `fit` принимает массив признаков `X` и вектор меток `y`

```python
rgen = np.random.RandomState(self.random_state)
```

Класс `RandomState` из модуля `random` библиотеки `numpy` предоставляет ряд методов генерации случайных чисел, взятых из различных распределений вероятностей. Здесь мы его инициализируем заранее заданным значением `random_state`, чтобы можно было тестировать гипотезы на воспроизводимых результатах.

```python
self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
```

Создаем nympy-массив со случайными весами по закону нормального распределения. Количество весов равно количеству признаков в датасете. Например, для 10 признаков и `random_state=1` данная строка кода сгенерирует такой массив:

```output
array([ 0.01624345, -0.00611756, -0.00528172, -0.01072969,  0.00865408,
       -0.02301539,  0.01744812, -0.00761207,  0.00319039, -0.0024937 ])
```

Здесь `loc` - это параметр, отвечающий за центр нормального распределения (среднее), у нас за такой центр выбрана точка 0, `scale` - это стандартное отклонение (разброс или ширина) распределения.

Можно убедиться, что веса генерируются по закону нормального распределения, если построить гистограмму. Например, если задать количество признаков 1000, то код построения гистограммы

```python
import numpy as np
import matplotlib.pyplot as plt

rgen = np.random.RandomState(1)
w_ = rgen.normal(loc=0.0, scale=0.01, size=1000)

plt.hist(w_, bins=20)
plt.show()
```

Нарисует следующую диаграмму, которая близка по форме к нормальному распределению:

![[Screenshot 2023-11-22 at 01.23.20.png]]

Следующая строка в методе `fit` инициализирует смещение (bias) нулем:
```python
self.b_ = np.float_(0.)
```

Далее инициализируем список, в котором будут храниться ошибки:

```python
self.errors_ = []
```

Затем идет цикл с количеством итераций, равным количеству эпох обучения. Эпоха - это один прогон всего датасета. За одну эпоху модель должна увидеть все образцы тренировочного набора данных.

```python
for _ in range(self.n_iter):
```

Внутри этого цикла на каждой итерации идет инициализация нулем переменной errors, которая будет накапливать количество ошибок модели за каждую эпоху.

```python
errors = 0
```

Следующий цикл отвечает за обновление весов модели после каждого образца:

```python
for xi, target in zip(X, y):
```

`xi` - массив признаков одного образца, а `target` - истинное значение (метка) образца.

```python
update = self.eta * (target - self.predict(xi))
```

update - это дельта, на которую нужно обновить веса модели. Из истинного значения класса образца вычитается предсказанное значение и эта разность умножается на коэффициент обучения. То есть, если класс предсказан верно (предсказание совпадает с меткой) разность будет равна нулю и веса обновлять не нужно. А если предсказан неверно, то в зависимости от того, что за ошибка, будет получено направление обновления весов (в сторону увеличения или уменьшения), а веса будут обновлены в эту сторону на значение лернинг рэйта.

```python
self.w_ += update * xi
self.b_ += update
```

Причем, здесь сразу все веса будут обновлены на одно и то же значение.

Если модель ошиблась на конкретном образце - добавляем переменной `errors` единицу. Ошибкой здесь считается любое значение переменной `update`, отличное от нуля. Выражение возвращает либо `True`, либо `False`, затем приводится к целому числу (что, на мой взгляд, даже лишее, `True` и `False` итак подклассы класса `int` и `True` автоматически приравнивается к 1, а `False` к 0)

```python
errors += int(update != 0.0)
```

Когда одна эпоха прошла накопленное количество ошибок `errors` добавляется в массив с ошибками по эпохам:

```python
self.errors_.append(errors)
```

И затем начинается следующая эпоха обучения.