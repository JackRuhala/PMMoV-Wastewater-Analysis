import streamlit as st
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import plotly.graph_objects as go
from scipy import stats
from sklearn.preprocessing import LabelEncoder

# Performance of laboratory extractions
WW_df = pd.read_csv(r'Wastewater data sheet')
         WW_df = WW_df.drop(columns = 'Unnamed: 0')

st.title('Variation of data explained by labritory factors')
st.write('''
         After an exploration of the PMMoV data, and possible environmental variables, it was found that a lot of “noise” was present.
         The theory is that there is either an environmental variable that has yet to be found causing fluctuations in the data, or there is a correlation between how the sample was collected or who was collecting it. 
         The given data has no information on environmental variables other than the variables recorded, could be causing a 10 fold change in PMMoV on a near daily basis. 
         What is provided is who was in the lab performing the extraction and how the sample was collected.
         Below is every recoreded extractor variation in PMMoV with flow rate influance removed
''')
Extractor_Preformance_df = WW_df.groupby('Extractor')['Log Residuals'].apply(list)
fig8 = px.box(WW_df, x='Extractor', y='Log Residuals')
st.plotly_chart(fig8)
st.write('''
         The Box plot tells us every extractor innate mean PMMoV recorded with flow rate influence removed.
         There is an inherent imbalance in the box plot as each individual extractor could have extracted 300 samples or 3 samples.
         Based on the box plot, no one extractor appears to have a mean PMMoV reading far greater than zero, 'the ideal mean'.
         We do not expect perfection in this metric as we don’t know the normal variance of PMMoV in wastewater, but we expect frequent extractors to all have similar means.
         The box plot gives the researchers confidence that PMMoV is not significantly skewed by individual extraction performance.
         The box plots also reinforce the training and the SOP dictated by the lab provide consistent results between lab members.
''')

st.write('''
         Next we want to compare the variance in PMMoV given diffrent sample collection methods given the variation of PMMoV due to flow rate is removed.
''')

fig9 = px.box(WW_df, x='Sample Type', y='Log Residuals')
st.plotly_chart(fig9)

st.write('''
         Like what we see in the extractor box plots, no significant difference in detected PMMoV variation was found between grab and composite samples.
         Again, we are not expecting tight distributions in our PMMoV counts since PMMoV is not constant, but one collection method should not have significantly more variation in PMMoV then the other.
         Both box plots conclude that variation in PMMoV when the influence of flow rate is removed is likely not caused by extractor performance or sample type.
''')
