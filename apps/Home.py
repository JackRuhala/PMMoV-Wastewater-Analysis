import streamlit as st
import pandas as pd
import numpy as np
from data.create_data import create_table

def app():
    st.title('Wastewater Epidemiology Enviromental Variable Analysis')

    st.write('''
    Welcome. This application explores data related to wastewater epidemiology. 
    What is wastewater epidemiology?  
    Wastewater epidemiology is a public health program that tracks diseases that can be detected in wastewater so inferences can be made about the severity and general location of an outbreak. 
    The methods used to track disease through wastewater might require some background knowledge in basic molecular biology central dogma to grasp what’s being discussed, but don’t worry, this app contains two sets of information. 
    One set of information will be targeted toward those with no background in molecular biology and the other set will contain more detailed information. 
    The page that explains the data for those with no molecular biology background will be oversimplified and will present some concepts incorrectly so that the bigger picture is better understood.
    ''')
    

    st.markdown("### Sample Data")
    df = create_table()
    st.write(df)

    st.write('Navigate to `Data Stats` page to visualize the data')
