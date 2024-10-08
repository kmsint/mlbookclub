parent: [[Глава 1. Введение в паттерны проектирования]]

tags: #OOP #book #pattern 

Для построения интерфейсов пользователя применяется тройка классов модель/представление/контроллер (Model/View/Controller — MVC).

**Модель** — это объект приложения. 

**Представление** — внешний вид приложения на экране. 

**Контроллер** описывает, как интерфейс реагирует на управляющие воздействия пользователя.

MVC отделяет представление от модели, устанавливая между ними протокол взаимодействия «подписка/уведомление».

Представление должно гарантировать, что оно отражает состояние модели. При каждом изменении внутренних данных модель уведомляет все зависящие от нее представления, в результате чего представление обновляет себя.

Такой подход позволяет присоединить к одной модели несколько представлений, обеспечив тем самым различные представления. Можно создать новое представление, не переписывая модель.

![[Screenshot 2024-07-26 at 00.04.00.png]]

На схеме показана одна модель и три представления. Разделение объектов происходит таким образом, что изменение одного отражается сразу на нескольких других, причем изменившийся объект не имеет информации о подробностях реализации других объектов. Этот более общий подход описывается паттерном проектирования [[Observer (наблюдатель)|наблюдатель]].

Еще одна особенность MVC заключается в том, что представления могут быть вложенными. Этот дизайн применим в ситуации, когда мы хотим иметь возможность группировать объекты и рассматривать группу как отдельный объект. Такой подход описывается паттерном [[Composite (компоновщик)|компоновщик]]. Он позволяет создавать иерархию классов, в которой некоторые подклассы определяют примитивные объекты (например, `Button` — кнопка), а другие — составные объекты (`CompositeView`), группирующие примитивы в более сложные структуры.

MVC позволяет также изменять реакцию представления на действия пользователя. MVC инкапсулирует механизм определения реакции в объекте `Controller`. Существует иерархия классов контроллеров, и это позволяет без труда создать новый контроллер как вариант уже существующего.

Представление пользуется экземпляром класса, производного от `Controller`, для реализации конкретной стратегии реагирования.

Отношение представление/контроллер — это пример паттерна проектирования [[Strategy (стратегия)|стратегия]]. Стратегия — это объект, представляющий алгоритм.

В MVC используются и другие паттерны проектирования, например [[Factory Method (фабричный метод)|фабричный метод]], позволяющий задать для представления класс контроллера по умолчанию, и [[Decorator (декоратор)|декоратор]] для добавления к представлению возможности прокрутки. Тем не менее, основные отношения в схеме MVC описываются паттернами [[Observer (наблюдатель)|наблюдатель]], [[Composite (компоновщик)|компоновщик]] и [[Strategy (стратегия)|стратегия]].