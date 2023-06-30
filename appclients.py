import pandas as pd
import streamlit as st
from PIL import Image
from model_ import open_data, preprocess_data, split_data, load_model_and_predict


def process_main_page():
    show_main_page()
    process_side_bar_inputs()


def show_main_page():
    image = Image.open('data/Без названия.jpg')

    st.set_page_config(
        layout="wide",
        initial_sidebar_state="auto",
        page_title="Clients",
        page_icon=image,

    )

    st.write(
        """
        # Классификация удовлетворенности клиента полетом
        Определяем, кто из пассажиров удовлетворен качеством обслуживания, а кто – нет.
        """
    )

    st.image(image)


def write_user_data(df):
    st.write("## Ваши данные")
    st.write(df)


def write_prediction(prediction, prediction_probas):
    st.write("## Предсказание")
    st.write(prediction)

    st.write("## Вероятность предсказания")
    st.write(prediction_probas)


def process_side_bar_inputs():
    st.sidebar.header('Заданные пользователем параметры')
    user_input_df = sidebar_input_features()

    train_df = open_data()
    train_X_df, _ = split_data(train_df)
    full_X_df = pd.concat((user_input_df, train_X_df), axis=0)
    preprocessed_X_df = preprocess_data(full_X_df, test=False)

    user_X_df = preprocessed_X_df[:1]
    write_user_data(user_X_df)

    prediction, prediction_probas = load_model_and_predict(user_X_df)
    write_prediction(prediction, prediction_probas)


def sidebar_input_features():
    gender = st.sidebar.selectbox("Пол", ("Мужской", "Женский"))
    age = st.sidebar.slider("Возраст", min_value=1, max_value=85, value=20,
                            step=1)
    customer_type = st.sidebar.selectbox("Тип потребителя", ("Лояльный клиент", "Нелояльный клиент"))
    type_of_travel = st.sidebar.selectbox("Тип поездки", ("Деловая поездка", "Личная поездка"))

    class_ = st.sidebar.selectbox("Класс", ("Бизнес", "Эко", "Эко плюс"))
    flight_distance = st.sidebar.slider("Дальность полета", min_value=1, max_value=89, value=20,
                            step=1)
    departure_delay_in_minutes = st.sidebar.slider("Задержка отправления", min_value=1, max_value=83, value=20,
                            step=1)
    arrival_delay_in_minutes = st.sidebar.slider("Задержка прибытия", min_value=1, max_value=84, value=20,
                            step=1)
    inflight_wifi_service = st.sidebar.slider("Вай-фай на борту", min_value=1, max_value=5, value=2,
                            step=1)
    departure_arrival_time_convenient  = st.sidebar.slider("Удобство времени отправления и прибытия", min_value=1, max_value=5, value=2,
                            step=1)
    ease_of_online_booking  = st.sidebar.slider("Удобство онлайн бронирования", min_value=1, max_value=5, value=2,
                            step=1)
    gate_location  = st.sidebar.slider("Расположение терминала", min_value=1, max_value=5, value=2,
                            step=1)
    food_and_drink  = st.sidebar.slider("Питание", min_value=1, max_value=5, value=2,
                            step=1)
    online_boarding  = st.sidebar.slider("Онлайн посадка", min_value=1, max_value=5, value=2,
                            step=1)
    seat_comfort  = st.sidebar.slider("Удобство сидений", min_value=1, max_value=5, value=2,
                            step=1)
    inflight_entertainment  = st.sidebar.slider("Развлечения на борту", min_value=1, max_value=5, value=2,
                            step=1)
    onboard_service  = st.sidebar.slider("Обслуживание посадки", min_value=1, max_value=5, value=2,
                            step=1)
    leg_room_service  = st.sidebar.slider("Место для ног", min_value=1, max_value=5, value=2,
                            step=1)
    baggage_handling  = st.sidebar.slider("Транспортировка багажа", min_value=1, max_value=5, value=2,
                            step=1)
    checkin_service  = st.sidebar.slider("Регистрация", min_value=1, max_value=5, value=2,
                            step=1)
    inflight_service  = st.sidebar.slider("Обслуживание на борту", min_value=1, max_value=5, value=2,
                            step=1)
    cleanliness  = st.sidebar.slider("Чистота", min_value=1, max_value=5, value=2,
                            step=1)


    translatetion = {
        "Мужской": "Male",
        "Женский": "Female",
        "Лояльный клиент": "Loyal Customer",
        "Нелояльный клиент": "disloyal Customer",
        "Деловая поездка": "Business travel",
        "Личная поездка": "Personal Travel",
        "Бизнес":  "Business",
        "Эко": "Eco",
        "Эко плюс": "Eco",
    }

    data = {
        "Gender": translatetion [gender],
        "Age": age,
        "Customer Type": translatetion [customer_type],
        "Type of Travel": translatetion [type_of_travel],
        "Class": translatetion [class_],
        "Flight Distance": flight_distance,
        "Departure Delay in Minutes": departure_delay_in_minutes,
        "Arrival Delay in Minutes": arrival_delay_in_minutes,
        "Inflight wifi service": inflight_wifi_service,
        "Departure/Arrival time convenient": departure_arrival_time_convenient,
        "Ease of Online booking": ease_of_online_booking,
        "Gate location": gate_location,
        "Food and drink": food_and_drink,
        "Online boarding": online_boarding,
        "Seat comfort": seat_comfort,
        "Inflight entertainment": inflight_entertainment,
        "On-board service": onboard_service,
        "Leg room service": leg_room_service,
        "Baggage handling": baggage_handling,
        "Checkin service": checkin_service,
        "Inflight service": inflight_service,
        "Cleanliness": cleanliness
    }

    df = pd.DataFrame(data, index=[0])

    return df

if __name__ == "__main__":
    process_main_page()


