parent: [[Встраивание ML-алгоритма в веб-приложение]]

tags: #mlbookclub #book_club #ml #python 

Конечно, не хочется каждый раз терять параметры тренированной модели сразу после того, как закрывается интерпретатор или ноутбук. Один из способов персистентного хранения обученных моделей - с помощью библиотеки `pickle`, позволяющей проводить сериализацию и десериализацию объектных структур python. В результате чего можно будет сохранять и загружать уже обученные модели