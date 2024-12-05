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
WW_df = WW_df.drop(columns = 'Unnamed: 0')
scaler = StandardScaler()

# Adress these issues later.
# WW_df_Standard['SNOW'] = WW_df['SNOW']
# WW_df_Standard['SNOW'] = scaler.fit_transform(WW_df_Standard['SNOW'])
# # WW_df_Standard['5'] = WW_df['Date']
# st.dataframe(WW_df_Standard)

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

# fig4 = px.scatter(WW_df_standerd, title = 'Sewer Flow Rate vs Enviroment', x='Date', y='FlowRate (MGD)')
# fig4.add_scatter(WW_df_standerd, x='Date',y='Discharge (ft^3/s)')
# st.plotly_chart(fig4)
# univariate graphs of pH of a system
fig5 = px.scatter(WW_df, title = 'Sewer Water pH by site', x='Date', y='pH', color ='Code', render_mode='svg')
st.plotly_chart(fig5)
# univariate graphs of PMMoV recorded in a system

fig5 = px.scatter(WW_df, title = 'PMMoV Gene Copys recorded in 100ml Sewer Water sample', x='Date', y='PMMoV (gc/ 100mL)', color ='Code', render_mode='svg')
st.plotly_chart(fig5)

###-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------###
# Building and environmental model for PMMoV
st.title('Building and environmental model for PMMoV')
st.write('''
         Below is an interactive figure of how our environmental data can be used to generate a model for PMMoV fluctuations
         The interactive plot generates a liner regression model of PMMoV fitted to the environmental variables of interest selected
         Model variables have already been determent but if you wish to see the models that are not further discussed feel free to come back and look at them here
'''
)

# Note The folowing code below calculates the optimal regretion line for the variables chosen.
# A function was added to allow the user to change the weights of the reggretion line which was supported by code writen by ChatGPT4.0 mini (12/4/2024)
# Appaun reveiew the code "works" but not as intended and any chage to the code will brake the whole streamlit page
# As of this comment(12/4/2024) I do not have time spend trouble shooting the code and need to focus on including more data.

WW_df_y = WW_df[['PMMoV (gc/ 100mL)', 'PMMoV Mean CT']]
WW_df_x = WW_df[['Discharge (ft^3/s)', 'FlowRate (MGD)','Temp', 'pH', 'Pellet Volume (ml)', 'PRCP (Rain fall in)']]
Code2 = st.selectbox("Select a Site Code", WW_df['Code'].unique())
column_y1 = st.selectbox("Select a Column for X-axis", WW_df_y.columns)
column_x1 = st.selectbox("Select a Column for X-axis", WW_df_x.columns)

# Display selected site code and column choices
st.write(f"Selected Site Code: {Code2}")
st.write(f"Y-axis Column: {column_y1}")
st.write(f"X-axis Column: {column_x1}")

# Filter the dataframe by selected site code
filtered_df = WW_df[WW_df['Code'] == Code2]
# Function to perform linear regression and calculate the best-fit line (min SSE)
def best_fit_line_slope(df, columnx, columny):
    if columnx not in df.columns or columny not in df.columns:
        st.error(f"Column {columnx} or {columny} not found in DataFrame.")
        return None, None
             
    # Drop NaN values from specified columns
    temp_df = df.dropna(subset=[columnx, columny, 'Date'])
    
    if temp_df.empty:
        st.error(f"Data after removing NaN values is empty. Please check the data.")
        return None, None
             
    # Get the X and Y values as numpy arrays
    X = np.array(temp_df[columnx], dtype=float)
    Y = np.array(temp_df[columny], dtype=float)
    Y = np.log10(Y)
   
    
    # Initial linear regression to get w1 and w0
    w1, w0, r, p, err = stats.linregress(X, Y)
    Y_predicted_min = w1 * X + w0
    SSE_mutiplyer = 2
    SSE_min = np.sum((Y - Y_predicted_min)**SSE_mutiplyer)
         
    # Generate ranges for w0 and w1 to minimize SSE
    w0_range = np.linspace(w0 * 0.25, w0 * 1.75, 200)
    w1_range = np.linspace(w1 * -10, w1 * 10, 200)

    # Initialize grid to store the sum of least squares (SSE) values
    SLS_grid = np.zeros((len(w0_range), len(w1_range)))

    # Loop through the range of w0 and w1 values to calculate SSE for each pair
    for i_idx, i in enumerate(w0_range):
        for j_idx, j in enumerate(w1_range):
            Y_predicted = j * X + i  # Predicted Y values based on current w0 and w1
            Sum_of_least_squares = np.sum((Y - Y_predicted)**SSE_mutiplyer)  # SSE for the current w0, w1 pair
            SLS_grid[i_idx, j_idx] = Sum_of_least_squares

    # Find the index of the minimum SSE in the grid
    min_SSE_index = np.unravel_index(np.argmin(SLS_grid), SLS_grid.shape)
    best_w0 = w0_range[min_SSE_index[0]]
    best_w1 = w1_range[min_SSE_index[1]]

    # The target SSE is 1.5 times the minimum SSE
    target_SSE = SSE_mutiplyer * SSE_min

    # Find the indices in the grid where SSE is approximately 2 times the minimum SSE
    tolerance = 0.05 * SSE_min  # Allow for small tolerance in SSE
    close_to_target_SSE = np.abs(SLS_grid - target_SSE) < tolerance

    # Get the coordinates of the points that are close to the target SSE
    indices = np.where(close_to_target_SSE)

    # Extract the corresponding w0 and w1 values
    selected_w0 = w0_range[indices[0]]
    selected_w1 = w1_range[indices[1]]

    if selected_w0.size == 0 or selected_w1.size == 0:
        st.error(f"No valid slope (w1) and intersept (w0) values found that satisfy the SSE condition.")
        return None, None, None, None, None, None, None, None
    # Find the longest line (the endpoints with the smallest and largest w0/w1)
    min_w0, max_w0 = selected_w0.min(), selected_w0.max()
    min_w1, max_w1 = selected_w1.min(), selected_w1.max()

    # Calculate the slope of the line connecting the endpoints with the target SSE
    slope = (max_w1 - min_w1) / (max_w0 - min_w0)

    intercept = min_w1 - slope * min_w0
    return min_w0, max_w0, min_w1, max_w1, slope, intercept, w0, w1

# Call the function to calculate the best-fit line parameters and slope
min_w0, max_w0, min_w1, max_w1, slope, intersept, w0, w1 = best_fit_line_slope(filtered_df, column_x1, column_y1)

st.write(f"The best-fit line parameters that minimize SSE are:")
st.write(f"The best-fit line Intersept (w0) = {w0}")   
# st.write(f"Plosible w0 (Intercept) range (SSE*2 max): {min_w0} to {max_w0}")
st.write(f"The best-fit line Slpoe (w1) = {w1}")
# st.write(f"Plosible w1 (Slope) range (SSE*2 max): {min_w1} to {max_w1}")
# st.write(f"The slope and intersept of the surface plot for the reggretion line of variables x and y with endpoints of the surface plot SSE close to 2x minimum SSE is: {slope} {intersept}")

if min_w0 is not None and min_w1 is not None:
    # Create a single slider for both w0 and w1
    slider_value = st.slider(
        "Adjust w0 and w1", 
        min_value=0.0, 
        max_value=100.0, 
        value=50.0
    )
    # Adjust w0 and w1 based on slider value
    # Mapping the slider to w0 and w1
    w0_adjusted = min_w0 + (slider_value / 100.0) * (max_w0 - min_w0)
    w1_adjusted = min_w1 + (slider_value / 100.0) * (max_w1 - min_w1)
    
    # Display the adjusted w0 and w1
    st.write(f"Adjusted w0 (Intercept): {w0_adjusted}")
    st.write(f"Adjusted w1 (Slope): {w1_adjusted}")
else:
    st.write("Please check the column selections and try again.")
y_temp = np.log10(filtered_df[column_y1])
fig6 = px.scatter(filtered_df, x=column_x1, y=y_temp, title=f"PMMoV liner regression model vs X {Code2}")
fig7 = px.scatter(filtered_df, x='Date', y=y_temp, title=f"PMMoV liner regression model vs Time {Code2}")
# Get the x-values from the filtered dataframe for plotting the regression line
x_values = filtered_df[column_x1].astype(float)

# Ensure w1_adjusted and w0_adjusted are scalars (if they're arrays, take the first element)
w1_adjusted = float(w1_adjusted)  # Ensure it's a scalar
w0_adjusted = float(w0_adjusted)  # Ensure it's a scalar

# Calculate the y-values of the regression line using the adjusted w1 and w0
y_values = (x_values * w1_adjusted) + w0_adjusted
SSE_adjusted = np.sum((np.log10(filtered_df[column_y1]) - y_values)**2)
st.write(f"Adjusted SSE = {SSE_adjusted}")
# Add the regression line as a new trace to the plot
fig6.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines', name='Regression Line', line=dict(color='red', width=2)))
fig7.add_trace(go.Scatter(x=filtered_df['Date'], y=y_values, mode='lines', name='Regression Line', line=dict(color='red', width=2)))
# # Display the plot using Streamlit
st.plotly_chart(fig6)
st.plotly_chart(fig7)







# Log multivariate liner regression code here

# Optional display of regression residual maps for w1 and w0 for univariate regression

# Print values of weights, the intercept, the sum of least squares, and  r^2 correlation value

st.write('''
         Based on carful observation of the liner models the best liner regression model involves fitting the data to flow rate and precipitation
         The variables used for liner regression were found to correlate with the method of sample collection
         Samples collected using a composite collected are most affected by flow rate.
         The flow rate of a system for composite samples is shown to be negatively correlated to the amount of PMMoV detected on a particular day.
         Faster than normal moving water through a system is suspected to flush out a system’s fecal mater and thus lower the PMMoV detected'
         Samples collected directly from the sewer were found to be greatly affected by heavy rainfall.
         The precipitation recorded over a 24 hour period positively correlates with PMMoV detected on the same day.
         heavy rainfall is suspected to greatly disturb a sewer environment causing lingering particulates to contaminate the grab sample more than normal, thus increasing the PMMoV detected
         Although flow rate and precipitation affect sample collection differently the environmental variable that greatly affects one sample has very little effect on the other sample.
         The effect each variable has on a sample site is reflected by the weights of the model given.
         Because of the dominance one variable in a sample has over the other, depending on how the sample is collected, both variables are included in one model for simplification of the models used across different sites.
''')

###-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------###
# Performance of laboratory extractions
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
         Simmiler to what we see in the extractor box plots, no significant diffrence in detected PMMoV variation was found between grab and composite samples.
         Again we are not expecting tight distributions in our PMMoV counts since PMMoV is not constent, but one collection method should not have significantly more variation in PMMoV then the other.
         Both box plots conclude that variation in PMMoV when the influance of flow rate is removed is likely not caused by extractor preformance or sample type.
''')

###-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------###
# Time potential dependency of residuals
st.title('Tempral analysis of PMMoV')

st.write('''
         One of the questions raised by the waste water lab was if PMMoV could be predicted.
         To Predict what PMMoV might be in the future, a ACF and PACF was preformed or PMMoV with the influance of flow rate removed.
''')
st.image('PMMoV_ACF_PACF.png')
st.write('''
         The ACF and PACF show that the PMMoV can not be predicted using an autocorilation function.
         The ACF and PACF suggest that ether tempral PMMoV is independent from one another where p=0 or a tempral prediction can be made where 0>p>1.
         If the autocorilation of PMMoV is 0>p>1 then the requency of sampling from bi-weekly to weekly is required.
         PMMoV data could be oversampled to artifishaly increase the sampling frequency to check for daily or weekly PMMoV cycles, but the sampling distribution of a single time point is needed to make a decent model.
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
         While an optimal amount of lags can be determaned through cross-corilation, that dose not mean the optimal lag is the best lag for the model.
         After testing a few lag models on diffrent sites, the best global lag for N1 is 8, 9, or 10.
         A lag of 8 or 10 is aproximently 1 month of time, so the data suggest that the N1 data can predict deaths due to COVID-19 one month in advance.
         Once a lag has been chosen, the N1 data has to be fit to the deaths data.
         After fitting N1 to deaths we test to see if PMMoV can improve our model residuals.
         As a reminder, to create an accurate model we need local COVID-19 related data so we expect our PMMoV model to only slighly improve our predictive power.
         We must also take into account the limitaions of N1 dettection using ddPCR.
         The N1 data stagnates as COVID-19 infections fall below 'a significant threshold'.
         N1 can only be called present in a sample if the 100ml sample count is aproximenly 2000 gc or above.
         If the N1 count data is below 2000 gc then we assume the sample has no N1.
         Because of the detection limitaions we can only use data where N1 is consistently above 2000.
         The data range for the scaled N1 graphs is limited to September 2023 and April 2024.
         The data for model fiting will also be limited to only site GR as GR is the only site that can be accuratly represented by the national COVID-19 data
         ''')

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
X_Flow = np.array(accuracy_test_df['FlowRate scaled (MGD)'])
X_PMMoV = np.array(accuracy_test_df['PMMoV scaled (gc/ 100mL)'])
Y_N1 = np.array(accuracy_test_df['N1 scaled Residuals Lag input'])
w1_PMMoV, w0_PMMoV, r_PMMoV, p_PMMoV, err_PMMoV = stats.linregress(X_PMMoV.astype(float), Y_N1.astype(float))
w1_Flow, w0_Flow, r_Flow, p_Flow, err_Flow = stats.linregress(X_Flow.astype(float), Y_N1.astype(float))
Y_predicted_PMMoV = w1_PMMoV * X_PMMoV.astype(float) + w0_PMMoV
Y_predicted_Flow = w1_Flow * X_Flow.astype(float) + w0_Flow
residuals_PMMoV = ((Y_N1 - Y_predicted_PMMoV) ** 2).sum()
residuals_Flow = ((Y_N1 - Y_predicted_Flow) ** 2).sum()

st.write("Explained variance in GR input lag N1 residuals using PMMoV")
st.write(f"Predicted Slope w1  = {w1_PMMoV:.4e}")
st.write(f"Predicted Intercept w0 = {w0_PMMoV:.4e}")
st.write(f"Person correlation r = {r_PMMoV:.4e}")
st.write(f"p_value = {p_PMMoV:.4e}")
st.write(f"Standerd error = {err_PMMoV:.4e}")
st.write(f"square sum of residuals with PMMoV= {residuals_PMMoV:.4e}")

fig13 = px.scatter(accuracy_test_df, x='Date', y='N1 scaled Residuals Lag input', title = 'liner reggretion of residual N1 lag to PMMoV')
fig13.add_trace(go.Scatter(x=accuracy_test_df['Date'], y=Y_predicted_PMMoV, mode='lines', name='Regression Line', line=dict(color='red', width=2)))
st.plotly_chart(fig13)


