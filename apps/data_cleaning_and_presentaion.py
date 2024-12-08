import streamlit as st
import pandas as pd
import 


WW_df = pd.read_csv(r'Wastewater data sheet')
WW_df = WW_df.drop(columns = 'Unnamed: 0')
scaler = StandardScaler()

def app():
         st.title('The Avalible Data')
         st.markdown('''
            If you have not already please read the introduction on wastewater epidemiology and PMMoV. If you dont have time to read the into or dont understand the context of what it means, here is a gross oversimplification of some of the more complicated data variables
            - PMMoV, Pi6, N1, N2, are all viruses.
            - gc/100ml = number of viruses per 100ml of collected sample.
            - gc/100ml = f(ct^-1).
            - PMMoV = Human fecal contamination.
            - Pi6 = Extraction control virus for N1 and N2.
            - N1 = N2 = COVID-19 (no specific variant) .
            ''')
         st.dataframe(WW_df, use_container_width=True)
         
         st.title('The Sorce of Data')
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
         st.title('Cleaning and Imputaion of Initial Data')
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
         Codes = WW_df['Code'].unique()
         user_input_1 = st.selectbox('Select A code', Codes)
         
# Discharge vs flow rate chart
         
# Box for user input 1
         Code_data = WW_df[WW_df['Code'] == user_input_1]
         
         Code_fr = np.array(Code_data['FlowRate (MGD)'])
         Code_dis = np.array(Code_data['Discharge (ft^3/s)'])
# Genrate stats for the liner regretion model after imputaion
         w1, w0, r, p, err = stats.linregress(Code_dis.astype(float),Code_fr.astype(float))
         
         st.write(f"Slope w1 ={w1}")
         st.write(f"Predicted Intercept w0 ={w0}")
         st.write(f"Predicted Pearson correlation coefficient r value ={r}")
         st.write(f"Predicted p-value ={p}")
         st.write(f"Predicted Standard error value ={err}")
         
         fig1 = px.scatter(Code_data, x='Discharge (ft^3/s)', y='FlowRate (MGD)', title=f"Discharge vs FlowRate for {user_input_1}")
         fig1.add_trace(go.Scatter(x=Code_dis, y=(Code_dis * w1 + w0), mode='lines', name='Regression Line', line=dict(color='red', width=2)))
         st.plotly_chart(fig1)
