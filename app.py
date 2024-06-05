import streamlit as st
import pandas as pd
import numpy as np
import joblib
from prediction import get_prediction

model = joblib.load(r'Model/model.joblib')

st.set_page_config(page_title ='Blueberry Yield Prediction', page_icon = '🫐', layout ='wide')
st.markdown("<h1 style = 'text-align: center;'>Blueberry Yield Prediction 🫐</h1>", unsafe_allow_html=True)

def main():
    with st.form('prediction_form'):

        st.subheader('Enter the input to following features:')
 
        clonesize = st.number_input("Clonesize: (m^2)", 10.0, 40.0, value='min', step=.01, format='%.2f')
        bumbles = st.number_input('Bumbles: (bees/m^2/min)', 0.0, 1.0, value='min', step=.01, format='%.2f')
        andrena = st.number_input('Andrena: (bees/m^2/min)', 0.0, 1.0, value='min', step=.01, format='%.2f')
        osmia = st.number_input('Osmia: (bees/m^2/min)', 0.0, 1.0, value='min', step=.01, format='%.2f')
        AverageOfUpperTRange = st.number_input('Average upper band daily air temperature: (℃)', 58.0, 80.0, value='min', step=.01, format='%.2f')
        AverageOfLowerTRange = st.number_input('Average lower band daily air temperature: (℃)', 40.0, 56.0, value='min', step=.01, format='%.2f')
        AverageRainingDays = st.number_input('Average of raining days: ', 0.0, 1.0, value='min', step=.01, format='%.2f')
 
        submit = st.form_submit_button('Predict Yield')

    if submit:

        data = np.array([clonesize, bumbles, andrena, osmia, AverageOfUpperTRange, AverageOfLowerTRange, AverageRainingDays]).reshape(1,-1)
        #data = [clonesize, bumbles, andrena, osmia, AverageOfUpperTRange, AverageOfLowerTRange, AverageRainingDays]
        pred = get_prediction(data, model)

        st.write(f'The predicted yield is: {pred:.2f}')


if __name__ == "__main__":
    main()
