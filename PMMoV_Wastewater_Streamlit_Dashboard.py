# packages
import streamlit as st
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import plotly.graph_objects as go
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
# from statsmodels.tsa.seasonal import seasonal_decompose
###-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------###

st.set_page_config(page_title='Kent County Michigan Viral Wastewater Analysis')

###-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------###
# Load in data

WW_df = pd.read_csv(r'Wastewater data sheet')

###-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------###

st.title('Analysis of Wastewater Epidemiology Variables')
st.write('Abstract: Tracking the origin of vectors of disease is one of the key rolls public health offcials are tasked with.'
         ' Wastewater epidemiology is a resent method used to track disease through non-confrontational means.'
         ' The main pitfall of wastewater disease tracking is the volatility of the sewer environment.'
         ' Here we look at data collected from the Grand Valley wastewater lab as well as data from Kent county weather,' 
         ' and geological stations, to better understand how the environment affects viral detection counts.')

###-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------###
# Introduction to wastewater epidemiology data
st.title('A Briefed Introduction to Wastewater Epidemiology Data')
st.write('Wastewater epidemiology is a new industry that aims to track vectors of disease through non-invasive means.'
         ' There are multiple methods of detecting disease through wastewater but the data we have uses a centrifugal extraction method with florescent PCR detection.'
         ' The workflow of wastewater epidemiology is as follows.'
         ' The prosess starts by collecting a sample of wastewater from a sewer.'
         ' A sample can be collected is one of two ways, by a composite extractor or a direct grab collection.'
         ' A composite extractor is a collection tank that collects a small portion every few min, over the course of a day, while a grab is just direct water collection out of a manhole.'
         ' Once the sample is collected it goes to a lab for processing.'
         ' The lab takes a portion of the sample collected and concentrates the sample by separating the water from the vector of disease.'
         ' Contention involves trapping viruses in a net made of polyethylene glycol (PEG), removing as much water from the PEG as possible, then extracting the collected viruses from the PEG.'
         ' Once the viruses are extracted, a small portion of the extraction is used to count the number of viruses in a sample.'
         ' Viral detection counts are recorded in two different forms depending on the detection method used.'
         ' The qPCR method records viral count as cycles (CT).'
         ' CT data is compared to a viral count standard which converts CT to gene copy data (gc) and gc is assumed to be an approximant equivalent of virus in a sample.'
         ' The other detection method id ddPCR which produces direct gc data.'
         ' There is more nuance and scientific reasoning behind the actual collection of data but knowing more than the basics should not be required for understanding the data.'
         ' In short, from start to finish, sample collection and viral detection is very involved and volatile so making notes of all variables between collection to detection is important for accurate data presentation.'
         ' The goal of this dashboard is to determine what factors impact viral detection in order to generate more accurate predictions of disease in a local population'
)
st.image('G_vs_C_sample_figure.png')
st.image('Extraction_Figure.png')
###-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------###
# The Key is PMMoV
st.title('What is PMMoV?')
st.write('A big part of this dashboard is understanding how the environment changes the detection of viruses in wastewater.'
         ' Studying the environments effect on the disease of interest in nearly imposable due to the true instance of a disease in a population is never truly known.'
         ' Instead of studying the effect of the environment on the disease directly, we can indirectly infer how the environment affects disease detection through a fecal matter variable.'
         ' Currently there are a number of proposed fecal matter controls in literature from detection of human genes in wastewater to caffein concentration of a sample.'
         ' The fecal matter control in our data is a plant virus called pepper mild monotilo virus or PMMoV for short.'
         ' PMMoV spreads from peppers, or processed pepper spices to human through consumption.'
         ' PMMoV endures the human digestive tract and is harmlessly expelled through our fecal matter where it enters the water and eventually infects more pepper plants.'
         ' Because PMMoV is expelled through human waste, PMMoV concentration is strongly positively correlated to human waste, and because pepper consumption is common in America most human waste contains PMMoV.'
         ' PMMoV also has the added benefit of being a virus.' 
         ' Although plant viruses have unique morphology compared to human viruses PMMoV is suspected of behaving similarly to viruses of interest during the collection, extraction and detection processes.'
         ' For all of the reasons listed above, PMMoV detection is interpreted as human fecal contamination data.'
         ' The higher the PMMoV counts, the more fecal matter in a sample, the higher the suspected count of disease.'
         ' If PMMoV counts change with environmental factors, then the suspected count of disease will positively corelate with the change in PMMoV.'
         ' The goal of this dashboard is to show the assumptions about PMMoVs direct positive correlation to disease are true.'
)

st.image('TMV.png', caption="This is an image of the Tobaco Mosaic virus, a close ansester of PMMoV, PMMoV and TMV are both rod shaped virused. Image was taken from https://www.semanticscholar.org/paper/The-physics-of-tobacco-mosaic-virus-and-virus-based-Alonso-G%C3%B3rzny/3177b81019a98aa9c2a17be46f325d1033f96f13")

###-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------###
st.title('Available Data')
st.markdown('''
            If you have not already please read the introduction on wastewater epidemiology and PMMoV. If you dont have time to read the into or dont understand the context of what it means, here is a gross oversimplification of some of the more complicated data variables
            - PMMoV, Pi6, N1, N2, are all viruses.
            - gc/100ml = number of viruses per 100ml of collected sample.
            - gc/100ml = f(ct^-1).
            - PMMoV = Human fecal contamination.
            - Pi6 = Extraction control virus for N1 and N2.
            - N1 = N2 = COVID-19 (no specific variant) .
            '''
)


# BEER_df = BEER_df.rename(columns={'Name': 'Name of Beer', 'Style_x':'Brewing Style', 'Style Color Mean': 'Style Color Mean (SRM)', 'Style Color Var':'Style Color Var (SRM)'})

# BEER_df = BEER_df.iloc[:,[19, 0, 3, 1, 2, 30, 5, 4, 26, 27, 6, 7, 28, 29, 20, 21, 22, 23, 24, 25, 9, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 31]]

st.dataframe(WW_df, use_container_width=True)

st.markdown('''
         Most of the data is publicly available online but all of the data is not available for download.
         All of the data is geographic contained within Kent county MI
         - N1 N2 data is published on the Michigan COVID-19 Wastewater Dashboard but dose not have a link to download the data
                  : https://www.michigan.gov/coronavirus/stats/wastewater-surveillance/wastewater-surveillance-for-covid-19/dashboard
         - The weather data was downloaded from NOAA, and collected from the Gerald ford weather station USC00202375,"EAST GRAND RAPIDS, MI US",
                  : https://www.weather.gov/wrh/climate?wfo=GRR
         - Daily discharge data can be downloaded from USGS grand rapids grand river monitoring station
                  : https://waterdata.usgs.gov/monitoring-location/04119000/#parameterCode=00065&period=P7D&showMedian=false
         - Publicly known deaths due to COVID-19 are published on the CDC website
                  : Place holder for link
         
         The remaining data presented in this dashboard was provided directly from the Grand Valley State University molecular monitoring lab.
         The lab data Is not found online but a link to the data used for this dashboard will be available on github
         This dashboard was given permission to use the data from the Grand Valley lab as well as make it public.
         As of today (12/2024) a publication of the data is in the works and more detailed information on how the data was collected will be presented in the publication
         '''
)
###-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------###
# Data cleaning of initial data
st.title('Data cleaning of initial data')
st.write('''
         Data that is available online for download and viewing is all complete and requires no imputation.
         Some of the weather pattern data was recorded in binary but never recorded the zero but instead reported as Na. 
         All Na in binary data were changed to 0.
         The lab data provided was not originaly presented as a df and had to be formatted into a CSV before imputation.
         The names of individual lab members performing analysis was encoded in the data to protect there identity.
         The remaining imputable factors are the temperature of the waste water, the flow rate of the sewer amd the pH of the waste water.
         '''
)
st.write('''
        Imputing flowrate and pH was done using the forward fill imputiontion method, where the reorded value of the day previous to the missing value replaces the missing value. 
        For the tempature of the sewer water, only one site 'CS' had tempature recorded.
        The assumption was made that because all sites share a close geological position to eachother, they would all have simmiler water tempatures, so the 'CS' tempature data was transferd over to the other sited and followed up with another forward fill.
        pH can be affected by tempature or flowrate of the water, so before imputing pH corilations betwen pH, water tempature, and flowrate were considerd.
        Assuming tempature to be a confounding variable, the effect of tempature on flow rate was removed from the flow rate data before comparing flow rate to pH.
        It was found that pH remaind stable in the sites recoreded regarless of tempature or flowrate, so pH was considerd to be independent from other variables and was imputed using the forward fill method.
        Not all sites had pH data recorded, and pH data can not be logicly transferd to other sites with no pH tempature.
        Last, Imputed variable was sewer water flow rate.
        For imputing flow rate, local ground water data was preferd but not found online.
        The next best thing to ground water data was data on the Grand River that flows through the County.
        The Discharge of the Grand River mesures how much water passess the ecological monitoring station per second per day.
        The daily Discharge of the Grand River was found to closely match the flow rate of the local sewer systems.
        Because trends in flow rate closely followed trends in discharge, the discharge of the Grand river was used to model what the flow rate would be on any given day.
        The model values of discharge vs flow rate are presented below.
         ''')
st.image('Imputed_GR_reggretion_map.png', caption= 'Example of how recoreded flowrate of a system corrilates with discharge of the Grand River before imputaion. A small anount of noise was added to the imputed data so the imputed flow rate dose not completly corrilate with discharge')
st.markdown(''' Sample stats of reggretion before imputaion

- Slope w1 = 0.00228
- Intercept w0 = 28.275
- Pearson correlation coefficient r value = 0.871
- p-value = 8.071e-37
- Standard error value = 0.000121
'''
)

user_input_1 = st.text_input('Enter Site Code Name:', 'Site Code')

# Discharge vs flow rate chart

# Box for user input 1
if user_input_1 in WW_df['Code'].values:
    st.write(user_input_1, ': Found')

    Code_data = WW_df[WW_df['Code'] == user_input_1]

    if not Code_data.empty:
    # Print value list
         Code_fr = np.array(Code_data['FlowRate (MGD)'])
         Code_dis = np.array(Code_data['Discharge (ft^3/s)'])
         # Genrate stats for the liner regretion model after imputaion
         w1, w0, r, p, err = stats.linregress(Code_dis.astype(float),Code_fr.astype(float))

         st.write(f"Slope w1 ={w1}")
         st.write(f"Predicted Intercept w0 ={w0}")
         st.write(f"Predicted Pearson correlation coefficient r value ={r}")
         st.write(f"Predicted p-value ={p}")
         st.write(f"Predicted Standard error value ={err}")

    else:
         st.write(user_input_1, ' : Code Not Found.')

    fig1 = px.scatter(Code_data, x='Discharge (ft^3/s)', y='FlowRate (MGD)', title=f"Discharge vs FlowRate for {user_input_1}")
    fig1.add_trace(go.Scatter(x=Code_dis, y=(Code_dis * w1 + w0), mode='lines', name='Regression Line', line=dict(color='red', width=2)))
    st.plotly_chart(fig1)
else:
    st.write(user_input_1, ' : Code Not Found')


###-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------###
# Understanding the environment of Kent Countys sewers
st.title('Understanding the Environment of Kent Countys sewers')
st.write('A sewer system is not isolated from the outside world, the system experiences dramatic changes along with the environment outside the environment'
         'Below, take some time to look at how the environment of a sewer system changes with the environment outside.'
)

fig2 = px.scatter(WW_df, title = 'Kent County Sewer Water Tempature', x='Date', y='Temp', render_mode='svg')
st.plotly_chart(fig2)
# univariate graphs of grand river discharge vs flow rate vs precipitation and snow melt
fig3 = px.scatter(WW_df, title = 'Sewer Flow Rate by site', x='Date', y='FlowRate (MGD)', color ='Code', render_mode='svg')
st.plotly_chart(fig3)
# univariate graphs of pH of a system
fig4 = px.scatter(WW_df, title = 'Sewer Water pH by site', x='Date', y='pH', color ='Code', render_mode='svg')
st.plotly_chart(fig4)
# univariate graphs of PMMoV recorded in a system
fig5 = px.scatter(WW_df, title = 'PMMoV Gene Copys recorded in 100ml Sewer Water sample', x='Date', y='PMMoV (gc/ 100mL)', color ='Code', render_mode='svg')
st.plotly_chart(fig5)

###-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------###
# Building and environmental model for PMMoV
st.title('Building and environmental model for PMMoV')
st.write('Below is an interactive figure of how our environmental data can be used to generate a model for PMMoV fluctuations'
         'The interactive plot generates a liner regression model of PMMoV fited to the environmental variables of interest selected'
         'Model variables have already been determent but if you wish to see the models that are not further discussed feel free to come back and look at them here'
)

# Log multivariate liner regression code here

# Optional display of regression residual maps for w1 and w0 for univariate regression

# Print values of weights, the intercept, the sum of least squares, and  r^2 correlation value

st.write('Based on carful observation of the liner models the best liner regression model involves fitting the data to flow rate and precipitation'
         'The variables used for liner regression were found to correlate with the method of sample collection'
         'Samples collected using a composite collected are most affected by flow rate.'
         'The flow rate of a system for composite samples is shown to be negatively correlated to the amount of PMMoV detected on a particular day.'
         'Faster than normal moving water through a system is suspected to flush out a systems fecal mater and thus lower the PMMoV detected'
         'Samples collected directly from the sewer were found to be grately affected by heavy rain fall.'
         'The precipitation recorded over a 24 hour period positively correlates with PMMoV detected on the same day.'
         'heavy rain fall is suspected to grately disturb a sewer environment causing lingering particulates to contaminate the grab sample more than normal, thus increasing the PMMoV detected'
         'Although flow rate and precipitation affect sample collection differently the environmental variable that gratly affect one sample has very little effect on the other sample.'
         'The effect each variable has on a sample site is reflected by the weights of the model given.'
         'Because of the dominance one variable in a sample has over the other, depending on how the sample is collected, both variables are included in one model for simplification of the models use across different sites'
)

###-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------###
# Performance of laboratory extractions

###-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------###
# Time potential dependency of residuals

###-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------###
