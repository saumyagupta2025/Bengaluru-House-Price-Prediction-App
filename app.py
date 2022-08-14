import pandas as pd
import numpy as np
import pickle
import streamlit as st
import json




model = pickle.load(open("banglore_home_prices_model.pickle", "rb"))
locations = json.load(open("columns.json"))
X = locations['data_columns']
locations_list = X[3:]

def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(X==location)[0]

    x = np.zeros(len(X))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return model.predict([x])[0]


def __main__():
    st.title("Bengaluru House Price Prediction")
    st.write("")
    st.image("02C.jpeg", width = 700)
    #st.write(locations_list)
    
    loc = st.selectbox("Enter the location:",locations_list) 
    sqft = st.number_input("Enter the area (in sq.ft): ", min_value = 100, max_value = 12000)
    sqft = float(sqft)
    bath = st.slider("Number of bathrooms: ", 1, 10)
    bath = float(bath)
    bhk = st.slider("BHK: ", 1, 10)

    sample = [loc, sqft, bath, bhk]
    sample = np.array(sample).reshape(1, -1)
    if st.button('Estimate Price'):

        pred = predict_price(loc, sqft, bath, bhk)
        pred = round(pred,2)
        output = "The estimated price of the property is " + str(pred) +  " Lakh"
        st.success(output)



   

__main__()