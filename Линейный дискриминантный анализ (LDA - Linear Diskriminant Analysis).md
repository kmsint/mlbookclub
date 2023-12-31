parent: [[Сжатие данных с помощью понижения размерности]]

tags: #ml #pca 

Линейный дискриминантный анализ (LDA) можно использовать как прием выделения признаков для повышения вычислительной продуктивности и уменьшения степени переобучения из-за "проклятия размерности" в нерегуляризированных моделях.

Метод LDA очень похож на метод [[Анализ главных компонентов (PCA - Principal Component Analysis)|PCA]] с тем различием, что PCA стремится отыскать ортогональные оси компонентов с максимальной дисперсией в наборе данных, а LDA ищет подпространство признаков, которое оптимизирует разделимость классов.

Оба анализа, РСА и LDA, являются приемами линейной трансформации, которые могут применяться для сокращения количества измерений в наборе данных; первый представляет собой алгоритм без учителя, тогда как второй - алгоритм с учителем.

На рисунке демонстрируется концепция LDA для двухклассовой задачи. Образцы из класса 1 изображены в виде окружностей, а образцы из класса 2 - в виде крестиков.

![[Screenshot 2023-12-16 at 15.47.47.png]]

[[Основные шаги при линейном дискриминантном анализе]]


