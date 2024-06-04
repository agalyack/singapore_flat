import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
import streamlit as st
import sklearn

st.set_page_config(
    page_title="Singapore  Resale Flat Prices Predicting",
    layout="wide",
    initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
    .header-container {
        background-color: purple; /* Purple background */
        padding: 20px; /* Increased padding */
        border-radius: 15px; /* Rounded corners */
        width: 100%; /* Make the container broader */
        margin-left: 10px; /* Align to left margin */
        margin-bottom: 20px; /* Reduce gap between header and subheader */
    }
    .header-text {
        font-family: Arial, sans-serif;
        color: white; /* White text color */
        margin: 0; /* Remove default margin */
        font-size: 24px; /* Increase font size */
    }
    </style>
    """,
    unsafe_allow_html=True
)    
st.markdown('<div class="header-container"><p class="header-text">Singapore  Resale Flat Prices Predicting</p></div>', unsafe_allow_html=True)

#columns under feature importance ['town', 'flat_type', 'storey_range', 'floor_area_sqm','lease_commence_date', 'year']
with open("r_model.pkl","rb") as rm:
    r_model_new=pickle.load(rm)
with open("labs1.pkl","rb") as lb1:
    lab_new1=pickle.load(lb1)
with open("labs2.pkl","rb") as lb2:
    lab_new2=pickle.load(lb2)
with open("labs3.pkl","rb") as lb3:
    lab_new3=pickle.load(lb3)

town_options=['TAMPINES','YISHUN','JURONG WEST','BEDOK','WOODLANDS','ANG MO KIO','HOUGANG','BUKIT BATOK','CHOA CHU KANG',
      'BUKIT MERAH','PASIR RIS','SENGKANG','TOA PAYOH','QUEENSTOWN','GEYLANG','CLEMENTI','BUKIT PANJANG','KALLANG/WHAMPOA',
      'JURONG EAST','SERANGOON','BISHAN','PUNGGOL','SEMBAWANG','MARINE PARADE','CENTRAL AREA','BUKIT TIMAH','LIM CHU KANG']
flat_type_options=['4 ROOM','3 ROOM','5 ROOM','EXECUTIVE','2 ROOM','1 ROOM','MULTI GENERATION','MULTI-GENERATION']
storey_range_options=['04 TO 06','07 TO 09','01 TO 03','10 TO 12','13 TO 15','16 TO 18','19 TO 21','22 TO 24','25 TO 27','01 TO 05',
              '06 TO 10','28 TO 30','11 TO 15','31 TO 33','34 TO 36','37 TO 39','16 TO 20','40 TO 42','21 TO 25','43 TO 45',
              '46 TO 48','26 TO 30','49 TO 51','36 TO 40','31 TO 35']
print("model loading")
tab1,tab2=st.tabs(["PREDICT SELLING PRICE","VISUALIZATIONS"])
with tab1:
    with st.form("my_form"):
       st.write("Flats Resale price prediction")
       town=st.selectbox("Town",town_options)
       flat_type=st.selectbox("Flat type",flat_type_options)
       storey_range=st.selectbox("Storey range",storey_range_options)
       floor_area_sqm=st.text_input("Floor area in SQM")
       lease_commence_date=st.text_input("Lease commence date")
       year=st.text_input("Year")

       submitted=st.form_submit_button("Predict")
       if submitted:
           columns=['town','flat_type','storey_range','floor_area_sqm','lease_commence_date','year']
           sample_data={
               'town':[town],
               'flat_type':[flat_type],
               'storey_range':[storey_range],
               'floor_area_sqm':[floor_area_sqm],
               'lease_commence_date':[lease_commence_date],
               'year':[year]               
           }
           df=pd.DataFrame(sample_data,columns=columns)
           # Apply label encoding
           df['town'] = lab_new1.transform(df['town'])
           df['flat_type'] = lab_new2.transform(df['flat_type'])
           df['storey_range'] = lab_new3.transform(df['storey_range'])
           pred=r_model_new.predict(df)
           sp=pred[0]
           st.markdown(f'### <div class="center-text">Predicted Resale Price = {sp}</div>', unsafe_allow_html=True)
            

