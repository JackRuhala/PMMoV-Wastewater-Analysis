import streamlit as st
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import plotly.graph_objects as go
from scipy import stats
from sklearn.preprocessing import LabelEncoder

def app():
# Performance of laboratory extractions
         WW_df = pd.read_csv(r'Wastewater data sheet')
         WW_df = WW_df.drop(columns = 'Unnamed: 0')

         st.title('Tempral analysis of PMMoV')
         
         st.write('''
                  One of the questions raised by the wastewater lab was if PMMoV could be predicted.
                  To Predict what PMMoV might be in the future, a ACF and PACF was preformed or PMMoV with the influence of flow rate removed.
         ''')
         st.image('PMMoV_ACF_PACF.png')
         st.write('''
                  The ACF and PACF show that the PMMoV cannot be predicted using an autocorrelation function.
                  The ACF and PACF suggest that either temporal PMMoV is independent from one another where p=0 or a temporal prediction can be made where 0>p>1.
                  If the autocorrelation of PMMoV is 0>p>1 then the frequency of sampling from bi-weekly to weekly is required.
                  PMMoV data could be oversampled to artificially increase the sampling frequency to check for daily or weekly PMMoV cycles, but the sampling distribution of a single time point is needed to make a decent model.
         ''')
         
         ###-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------###
         st.title('Tempral analysis of N1 controled for PMMoV')
         
         st.write('''
                  All the analysis on PMMoV and the environment is needed to tell us something about the fecal contamination of a sample.
                  The idea is that the more fecal contaminated the sample, the more fecal pathogens will be detected.
                  One such fecal pathogen is COVID-19. 
                  The detection of COVID-19 gene N1 and N2 in fecal matter has given public health officials a way to predict COVID-19 infections before signs of an outbreak appear in the population.
                  The N1 and N2 counts collected from wastewater closely follow the reported infections of COVID-19 a few weeks before they are reported.
                  To gauge how well fecal N1 predicts COVID-19 infections, the locally known number of infected people is needed to make an accurate prediction, however, since March of 2023, many of the COVID-19 dashboards stopped reporting data.
                  Instead of using locally known COVID-19 infections to compare to our N1 data, we will use the National reported deaths due to COVID-19.
                  While national data is not the best to use for this type of data, if we look at data from a site with a large population and service area the site N1 should come close to the nationally reported number.
         ''')
         st.write('''
                  To compare how well N1 compares to deaths due to COVID-19, first the lag between N1 and deaths has to be found.
                  ''')
         # set lag range
         Code3 = st.selectbox("Select a Site Code", WW_df['Code'].unique(), key="Lag_box")
         st.write(f"Selected Site Code: {Code3}")
         filtered_lag_df1 = WW_df[WW_df['Code'] == Code3]
         
         max_lag = 20
         cross_corr = np.correlate(filtered_lag_df1['BiWeekly Deaths'], filtered_lag_df1['N1'], mode='full')
         lags = np.arange(-max_lag, max_lag + 1)
         start_idx = len(filtered_lag_df1) - 1 - max_lag
         end_idx = len(filtered_lag_df1) - 1 + max_lag + 1
         cross_corr_lagged = cross_corr[start_idx:end_idx]
         
         fig10 = px.line(filtered_lag_df1, x=lags, y=cross_corr_lagged)
         st.plotly_chart(fig10)
         # pull the lag with the highest corrilation
         optimal_p = lags[np.argmax(cross_corr_lagged)]
         st.write(f'The sugested lag is {optimal_p}')
         
         user_input_lag = st.text_input('Enter a number of lags', '0')
         user_input_lag = int(user_input_lag)
         filtered_lag_df1['N1_Lagged'] = filtered_lag_df1['N1'].shift(user_input_lag)
         scaler = StandardScaler()
         
         # Fit and transform the 'BiWeekly Deaths' and 'N1_Lagged' columns
         filtered_lag_df1['BiWeekly Deaths scaled'] = scaler.fit_transform(filtered_lag_df1[['BiWeekly Deaths']])
         filtered_lag_df1['N1_Lagged_scaled'] = scaler.fit_transform(filtered_lag_df1[['N1_Lagged']])
         
         fig11 = px.scatter(filtered_lag_df1, x='Date', y='BiWeekly Deaths scaled', color_discrete_sequence=['blue'])
         fig11.add_scatter(x=filtered_lag_df1['Date'], y=filtered_lag_df1['N1_Lagged_scaled'], mode='markers', marker=dict(color='red'))
         fig11.data[0].name = 'BiWeekly Deaths scaled'
         fig11.data[1].name = 'N1_Lagged_scaled'
         st.plotly_chart(fig11)
         
         st.write('''
                  While an optimal number of lags can be determined through cross-correlation, that does not mean the optimal lag is the best lag for the model.
                  After testing a few lag models on different sites, the best global lag for N1 is 8, 9, or 10.
                  A lag of 8 or 10 is approximately 1 month of time, so the data suggest that the N1 data can predict deaths due to COVID-19 one month in advance.
                  Once a lag has been chosen, the N1 data must be fit to the deaths data.
                  After fitting N1 to deaths we test to see if PMMoV can improve our model residuals.
                  As a reminder, to create an accurate model we need local COVID-19 related data so we expect our PMMoV model to only slightly improve our predictive power.
                  We must also consider the limitations of N1 detection using ddPCR.
                  The N1 data stagnates as COVID-19 infections fall below 'a significant threshold'.
                  N1 can only be called present in a sample if the 100ml sample count is approximately 2000 gc or above.
                  If the N1 count data is below 2000 gc then we assume the sample has no N1.
                  Because of the detection limitations we can only use data where N1 is consistently above 2000.
                  The data range for the scaled N1 graphs is limited to September 2023 and April 2024.
                  The data for model fitting will also be limited to only site GR as GR is the only site that can be accurately represented by the national COVID-19 data
                  ''')
         
         # Cross correlation code was written by ChatGPT4.0 mini but manually changed and checked over time to fit the streamlit app
         accuracy_test_df = WW_df[WW_df['Code'] == 'GR']
         # user inputs number of lag steps for plot
         user_input_GR_lag = st.text_input('Enter a number of lags for site GR', '8')
         user_input_GR_lag = int(user_input_GR_lag)
         accuracy_test_df = accuracy_test_df.loc[(accuracy_test_df['Date'] >= '2023-09-01') & (accuracy_test_df['Date'] <= '2024-04-01')]
         # remove columns with missing data
         accuracy_test_df = accuracy_test_df.dropna(subset=['PMMoV (gc/ 100mL)', 'FlowRate (MGD)', 'N1','BiWeekly Deaths', 'Date'])
         accuracy_test_df['BiWeekly Deaths scaled'] = scaler.fit_transform(accuracy_test_df[['BiWeekly Deaths']])
         accuracy_test_df['N1 Lagged scaled'] = scaler.fit_transform(accuracy_test_df[['N1']])
         accuracy_test_df['FlowRate scaled (MGD)'] = scaler.fit_transform(accuracy_test_df[['FlowRate (MGD)']])
         accuracy_test_df['PMMoV scaled (gc/ 100mL)'] = scaler.fit_transform(np.log10(accuracy_test_df[['PMMoV (gc/ 100mL)']]))
         # Inverse lag the Biweekly lag data to reduce the amount of data that need to be shifted for the liner reggretion to work.
         accuracy_test_df['BiWeekly Deaths scaled input'] = accuracy_test_df['BiWeekly Deaths scaled'].shift(-1*(user_input_GR_lag))
         accuracy_test_df['N1 scaled Residuals Lag input'] = accuracy_test_df['N1 Lagged scaled'] - accuracy_test_df['BiWeekly Deaths scaled input']
         # add flowrate to 'N1 scaled Residuals Lag input' because flowrate is inverly corralated to N1
         accuracy_test_df['N1 flowrate scaled Residuals Lag input'] = accuracy_test_df['N1 scaled Residuals Lag input'] + accuracy_test_df['FlowRate scaled (MGD)']
         accuracy_test_df['N1 PMMoV scaled Residuals Lag input'] = accuracy_test_df['N1 scaled Residuals Lag input'] - accuracy_test_df['PMMoV scaled (gc/ 100mL)']
         
         SSE_N1_input_lag =(accuracy_test_df['N1 scaled Residuals Lag input']**2).sum()
         SSE_N1_flow_input_lag = np.sum(accuracy_test_df['N1 flowrate scaled Residuals Lag input']**2)
         SSE_N1_PMMoV_input_lag = np.sum(accuracy_test_df['N1 PMMoV scaled Residuals Lag input']**2)
         
         fig12 = px.line(accuracy_test_df, x='Date', y='N1 scaled Residuals Lag input', title = 'GR N1 data scaled Residuals to Lag input fitted to national COVID-19 death data')
         st.plotly_chart(fig12)
         
         st.write(f'SSE for N1 input lag: {SSE_N1_input_lag}')
         
         # Get the flow rate and discharge values as numpy arrays
         Y_N1 = np.array(accuracy_test_df['N1 scaled Residuals Lag input'])
         Y_N1 = Y_N1.astype(float)
         # remove missing values from the Y array
         mask = ~np.isnan(Y_N1)
         Y_N1 = Y_N1[mask]
         # ajust the shape of the X arrays to match the Y
         X_Flow = np.array(accuracy_test_df['FlowRate scaled (MGD)'])[mask]
         X_Flow = X_Flow.astype(float)
         X_PMMoV = np.array(accuracy_test_df['PMMoV scaled (gc/ 100mL)'])[mask]
         X_PMMoV = X_PMMoV.astype(float)
         
         # generate liner reggretion stats
         w1_PMMoV, w0_PMMoV, r_PMMoV, p_PMMoV, err_PMMoV = stats.linregress(X_PMMoV, Y_N1)
         w1_Flow, w0_Flow, r_Flow, p_Flow, err_Flow = stats.linregress(X_Flow, Y_N1)
         Y_predicted_PMMoV = w1_PMMoV * X_PMMoV + w0_PMMoV
         Y_predicted_Flow = w1_Flow * X_Flow + w0_Flow
         residuals_PMMoV = np.sum((Y_N1 - Y_predicted_PMMoV) ** 2)
         residuals_Flow = np.sum((Y_N1 - Y_predicted_Flow) ** 2)
         
         st.write("Explained variance in GR input lag N1 residuals using PMMoV")
         st.write(f"Predicted Slope w1  = {w1_PMMoV}")
         st.write(f"Predicted Intercept w0 = {w0_PMMoV}")
         st.write(f"Person correlation r = {r_PMMoV}")
         st.write(f"p_value = {p_PMMoV}")
         st.write(f"Standerd error = {err_PMMoV}")
         st.write(f"square sum of residuals with PMMoV= {residuals_PMMoV}")
         
         fig13 = px.scatter(accuracy_test_df, x='Date', y='N1 scaled Residuals Lag input', title = 'liner regression of residual N1 lag to PMMoV')
         fig13.add_trace(go.Scatter(x=accuracy_test_df['Date'], y=Y_predicted_PMMoV, mode='lines', name='Regression Line', line=dict(color='red', width=2)))
         st.plotly_chart(fig13)
         
         st.write("Explained variance in GR input lag N1 residuals using Flow rate")
         st.write(f"Predicted Slope w1  = {w1_Flow}")
         st.write(f"Predicted Intercept w0 = {w0_Flow}")
         st.write(f"Person correlation r = {r_Flow}")
         st.write(f"p_value = {p_Flow}")
         st.write(f"Standerd error = {err_Flow}")
         st.write(f"square sum of residuals with PMMoV= {residuals_Flow}")
         
         fig14 = px.scatter(accuracy_test_df, x='Date', y='N1 scaled Residuals Lag input', title = 'liner regression of residual N1 lag to Flow rate')
         fig14.add_trace(go.Scatter(x=accuracy_test_df['Date'], y=Y_predicted_Flow, mode='lines', name='Regression Line', line=dict(color='red', width=2)))
         st.plotly_chart(fig14)
