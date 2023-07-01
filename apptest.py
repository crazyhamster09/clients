import pandas as pd
import streamlit as st
from PIL import Image

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
