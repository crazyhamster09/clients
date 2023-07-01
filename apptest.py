import pandas as pd
import streamlit as st
from PIL import Image

image = Image.open('data/Без названия.jpg')
st.write(
        """
        # Классификация удовлетворенности клиента полетом
        Определяем, кто из пассажиров удовлетворен качеством обслуживания, а кто – нет.
        """
    )

    st.image(image)
