#!/opt/homebrew/bin/python3

import numpy as np
import pandas as pd
import streamlit as st
import pickle
import sklearn as skl

#load_model
lr=pickle.load(open('./lr.pkl','rb'))

#setup_webpage
st.set_page_config(page_title='Salary Pred',
    page_icon='random',
    )

st.title("Salary Predictor")
st.header("Enter your experience in years:")

#setup_input_field
years=st.number_input('Years: ',min_value=0,max_value=12,step=1)

#setup_output_field
if st.button('Predict Salary'):
    sal=lr.predict([[years]])
    st.success(f'Expected Salary: {sal[0]:.2f}')

