# Задача прогноза удовлетворенности клиента полетом

Проект представления модели машинного обучения в виде веб-приложения с помощью фреймворка [Streamlit](https://www.streamlit.io/). Модель построена на данных датасета ["Клиенты авиакомпании"](https://github.com/crazyhamster09/clients/blob/main/data/clients.csv). Приложение представлено [здесь](https://clients-7tpl0t8vtfr.streamlit.app/).

## Файлы
- `apptest.py`: файл приложения
- `model_.py`: файл модели
- `clients.csv`: датасет
- `requirements.txt`: требования к установленным пакетам

## Описание датасетa "Клиенты авиакомпании"

Датасет содержит информацию о клиентах некоторой авиакомпании.

*Целевая переменная* (таргет) – `satisfaction` (удовлетворенность клиента полетом), бинарная (*satisfied* или *neutral or dissatisfied*)

### Признаки
- `Gender` (categorical: *Male* или *Female*): пол клиента

- `Age` (numeric, int): количество полных лет

- `Customer Type` (categorical: *Loyal Customer* или *disloyal Customer*): лоялен ли клиент авиакомпании?

- `Type of Travel` (categorical: *Business travel* или *Personal Travel*): тип поездки

- `Class` (categorical: *Business*, *Eco* или *Eco Plus*): класс обслуживания в самолете

- `Flight Distance` (numeric, int): дальность перелета (в милях)

- `Departure Delay in Minutes` (numeric, int): задержка отправления (неотрицательная)

- `Arrival Delay in Minutes` (numeric, int): задержка прибытия (неотрицательная)

Признаки, перечисленные ниже, являются числовыми. По смыслу они категориальные: клиент ставил оценку от 1-го до 5-ти включительно. Есть выбросы!

- `Inflight wifi service` (categorical, int): оценка клиентом интернета на борту

- `Departure/Arrival time convenient` (categorical, int): оценка клиентом удобство времени прилета и вылета

- `Ease of Online booking` (categorical, int): оценка клиентом удобства онлайн-бронирования

- `Gate location` (categorical, int): оценка клиентом расположения выхода на посадку в аэропорту

- `Food and drink` (categorical, int): оценка клиентом еды и напитков на борту

- `Online boarding` (categorical, int): оценка клиентом выбора места в самолете

- `Seat comfort` (categorical, int): оценка клиентом удобства сиденья

- `Inflight entertainment` (categorical, int): оценка клиентом развлечений на борту

- `On-board service` (categorical, int): оценка клиентом обслуживания на борту

- `Leg room service` (categorical, int): оценка клиентом места в ногах на борту

- `Baggage handling` (categorical, int): оценка клиентом обращения с багажом

- `Checkin service` (categorical, int): оценка клиентом регистрации на рейс
- `Inflight service` (categorical, int): оценка клиентом обслуживания на борту
- `Cleanliness` (categorical, int): оценка клиентом чистоты на борту

## Обработка данных и построение модели
В данных были убраны выбросы.
Для построения модели была использована линейная регрессия.
