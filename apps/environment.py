import streamlit as st
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import plotly.graph_objects as go
from scipy import stats
from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LinearRegression

def app():
         WW_df = pd.read_csv(r'Wastewater data sheet')
         WW_df = WW_df.drop(columns = 'Unnamed: 0')
         # scaler = StandardScaler()
         # Understanding the environment of Kent Countys sewers
         st.title('Understanding the Environment of Kent Countys sewers')
         st.write('''
                  A sewer system is not isolated from the outside world, the system experiences dramatic changes along with the environment outside the environment. 
                  The effect the enviroment has on the sewer enviroment is mostly determend by the design of the sewer itself.
                  Sewers are build as a network of pipes that feed into incresingly larger pipes, the larger the pipe the larger the sample population of an area.
                  Sewers can also be combined with storm water and older sewer systems show cracks in the pipes that let ground water infiltrate the system.
                  Below, take some time to look at how the environment of a sewer system changes with the environment outside.
                  '''
         )

         # loacl water temp graph
         fig1 = px.scatter(WW_df, title = 'Kent County Sewer Water Tempature', x='Date', y='Temp', render_mode='svg')
         st.plotly_chart(fig1)

         # local snow depth graph
         fig2 = px.scatter(WW_df, title = 'local Snow depth in inches', x='Date', y='SNWD (Snow Depth)', render_mode='svg')
         st.plotly_chart(fig2)
         st.write('''
                  Note: A lot if snow data from winter 2023 was cut to fit the rest of the data table.
                  ''')
         # local Precipitaion graph
         fig3 = px.scatter(WW_df, title = 'local Precipitaion in inches', x='Date', y='PRCP (Rain fall in)', render_mode='svg')
         st.plotly_chart(fig3)
         
         # Grand River graph
         fig4 = px.scatter(WW_df, title = 'Discharge of the Grand River (ft^3/s)', x='Date', y='Discharge (ft^3/s)')
         st.plotly_chart(fig4)
         st.write('''
                  Note: When looking at how discharge of the Grand River is affected by weather patterns, remember the grand river spans well outside the local area of intrest. 
                  Because of the grand rivers vast area, imagine how much more water enters the river when snow melts on frozen ground compred to a hevey storm on dry ground.
                  ''')
         # Sewer flow rate graph
         fig5 = px.scatter(WW_df, title = 'Sewer Flow Rate by site', x='Date', y='FlowRate (MGD)', color ='Code', render_mode='svg')
         st.plotly_chart(fig5)
         
         # pH of a system graph
         fig6 = px.scatter(WW_df, title = 'Sewer Water pH by site', x='Date', y='pH', color ='Code', render_mode='svg')
         st.plotly_chart(fig6)
         
         # graphs of PMMoV recorded in a system
         fig7 = px.scatter(WW_df, title = 'PMMoV Gene Copys recorded in 100ml Sewer Water sample', x='Date', y='PMMoV (gc/ 100mL)', color ='Code', render_mode='svg')
         st.plotly_chart(fig7)
