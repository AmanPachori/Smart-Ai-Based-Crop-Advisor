import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import warnings

modelfile = "./model_Logistic_Regression.pkl"

def load_model(modelfile):
    loaded_model = pickle.load(open(modelfile, 'rb'))
    return loaded_model

def main():
    # title
    st.set_page_config(page_title="Crop Advisor", page_icon="🧑‍🌾", layout='centered', initial_sidebar_state="collapsed")
    html_temp = """
    <div>
    <h1 style="color:MEDIUMSEAGREEN;text-align:left;">Krishak</h1>
    <h2 style="color:MEDIUMSEAGREEN;text-align:left;">Smart Crop Recommendation Platform</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 2])

    with col1:
        with st.expander(" ℹ️ Information", expanded=True):
            st.write("""
            ## About Krishak
Krishak, is a smart crop recommendation platform. It helps farmers by recommending the most suitable crop based on a variety of factors such as soil nutrition and environmental conditions. 
The key feature of this platform is the focus on local adaptation, which means that the recommendations are tailored to the local environment.
This is achieved by integrating secondary data available on web resources with the primary data collected from farmers to train the platform's ML models.            """)
        with st.expander(" ℹ️ Information", expanded=True):
             st.write('''
             ## How it works❓ 
            Fill in all the parameters, and the machine learning model will make predictions about the most appropriate crops to cultivate on a specific farm, considering a range of factors.
            ''')

    with col2:
        st.subheader("Find out the most suitable crop to grow in your farm🧑‍🌾")
        N = st.number_input("Nitrogen", 1, 10000)
        P = st.number_input("Phosphorus", 1, 10000)
        K = st.number_input("Potassium", 1, 10000)
        temp = st.number_input("Temperature", 0.0, 100000.0)
        humidity = st.number_input("Humidity in %", 0.0, 100000.0)
        ph = st.number_input("pH", 0.0, 100000.0)
        rainfall = st.number_input("Rainfall in mm", 0.0, 100000.0)

        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        if st.button('Predict'):
            loaded_model = load_model('model_Logistic_Regression.pkl')
            prediction = loaded_model.predict(single_pred)
            st.write('''
            ## Results 🔍 
            ''')
            st.success(f"{prediction.item().title()} are recommended by the A.I for your farm.")

    st.warning("Note: This A.I application is for educational/demo purposes only and cannot be relied upon.")
    hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    </style>
    """
    st.markdown(hide_menu_style, unsafe_allow_html=True)

if __name__ == '__main__':
    main()