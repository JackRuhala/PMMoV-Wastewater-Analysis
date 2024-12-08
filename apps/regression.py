import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import plotly.graph_objects as go
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LinearRegression
def app():
         WW_df = pd.read_csv(r'Wastewater data sheet')
         WW_df = WW_df.drop(columns = 'Unnamed: 0')
         # Building and environmental model for PMMoV
         st.title('Building and environmental model for PMMoV')
         st.write('''
                  Below is a figure of correlation variables within the table. 
                  The correlations are determined using a correlation matrix of normally scaled data.
                  For those who might not understand correlation matrixes, a 1 or -1 means the two intersecting columns are correlated and a 0 means the two intersecting columns are not correlated.
                  The negative and positive values determine the type of correlation. 
                  Negative values are negatively correlated and positive values are positively correlated.
                  The correlation matrix only estimates the correlation between two variables and becomes less accurate, the more a variable is impacted by other variables, so take the estimated correlations with a grain of salt.
                  ''')
         # Generate a Cov matrix
         # Select a site
         Code1 = st.selectbox("Select a Site Code for the corrilation matrix", WW_df['Code'].unique(), key="cov_box")
         # filter out site spicifc data
         filtered_df1 = WW_df[WW_df['Code'] == Code1]
         
         filtered_df1 = filtered_df1.drop(columns= ['Date','Code','BiWeekly Deaths','qPCR Operator','Date tested','Lag date','Sample Type','Log Residuals'])
         if  filtered_df1['pH'].isna().sum() > 10:
                  filtered_df1 = filtered_df1.drop(columns= ['pH'])
         filtered_df1 = filtered_df1.dropna()
         
         # Scale the remaining data
         filtered_scaled = StandardScaler().fit_transform(filtered_df1)
         filtered_cov = np.cov(filtered_scaled.T)
         
         # re introduce column names since they got lost after scaling the data
         filtered_cov_heatmap = pd.DataFrame(filtered_cov, columns = filtered_df1.columns)
         fig = plt.figure(figsize=(20,20))
         sns.heatmap(filtered_cov_heatmap, annot=True, cmap = 'icefire', xticklabels=filtered_cov_heatmap.columns, yticklabels=filtered_cov_heatmap.columns)
         plt.title('Corrilation matrix heatmap')
         st.pyplot(fig)
         
         # Note The folowing code below calculates the optimal regretion line for the variables chosen above, from the select boxs.
         # A function was added to allow the user to change the weights of the reggretion line which was supported by code writen by ChatGPT4.0 mini (12/4/2024)
         st.write('''
                  Here we are trying to find the liner relationship between our PMMoV gene counts (gc) per 100ml sample and our environmental variables.
                  PMMoV (gc/100ml) is presented in log10 form for two reasons, one its much easier to see changes in PMMoV in log form, and two, PMMoV is normally distributed around the regression line in log form as was determined by QQ-plots.
                  The reggertion line is inishaly calculated using formula...
                  ''')
         st.latex(r'''SSE = \sum^n_{t=1} (y_t - (w_1x_t + w_0))^2''')
         st.write('''
                  Where the goal is to minimize the value of SSE (sum of least squres) to get the line that fits the data the "best".
                  To minimize the SSE, the w1 and w0 for minimum loss must be found.
                  To find the best w1 and w0 for the minimum SSE we use the sumrized formula...
         ''')
         st.latex(r'''Y = XW ''')
         st.write('''
                  Where X and Y are matries of our x and y variable data respectivly, and W is a matrix containing w0 and w1.
                  Once w0 and w1 are applyed to our data, we cant garentie that the best fit line is the line that represents the variable relationship in reality.
                  Our data has meny unexplored variances, so we need to find combinations of w0 and w1 that genrate SSE with close to minimal loss and compare the lines of minimal loss to each other.
                  Below, select an site location and an enviromental variable.
                  The application will use the data avalibe to genrate an optimal regression line using the SSE loss formula.
                  Once the optimal SSE is found, the app will look for sets of w1 and w0 with SSEs close to the minimum SSE, this is called genrating a surface map.
                  Once pairs of w0 and w1 are found, the app will find the most "significant" w0 and w1 pairs and load them onto the slider bar below.
                  To adjust the regression line move the slider back and forth.
                  The origanl, optiomaly best fit line is set in the middle of the slider at 50 and its values are recorded below.
                  There is no one best fit line for this data and more research needs to be done to narrow down the options for best fit line.
         ''')
         WW_df_y = WW_df[['PMMoV (gc/ 100mL)']]
         WW_df_x = WW_df[['Discharge (ft^3/s)', 'FlowRate (MGD)','Temp', 'pH', 'Pellet Volume (ml)', 'PRCP (Rain fall in)']]
         Code2 = st.selectbox("Select a Site Code for regretion", WW_df['Code'].unique(), key="reg_box")
         column_y1 = 'PMMoV (gc/ 100mL)'
         column_x1 = st.selectbox("Select a Column for X-axis", WW_df_x.columns)
         
         # Display selected site code and column choices
         st.write(f"Selected Site Code: {Code2}")
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
             # check to make sure temp_df is not empty
             if temp_df.empty:
                 st.error(f"Data after removing NaN values is empty. Please check the data.")
                 return None, None
                      
             # Get the X and Y values as numpy arrays
             # change array data type to a float
             X = np.array(temp_df[columnx], dtype=float)
             Y = np.array(temp_df[columny], dtype=float)
             # Take log 10 of PMMoV gc counts per 100ml sample for right skew data adjustment of normaly distributed data fond by QQ-ploting
             Y = np.log10(Y)
            
             # calculate w1 and w0 of the regression
             w1, w0, r, p, err = stats.linregress(X, Y)
             Y_predicted_min = w1 * X + w0
             SSE_range_mutiplyer = 2
             SSE_min = np.sum((Y - Y_predicted_min)**2)
                  
             # run a grid search for posible w0 and w1 to
             w0_range = np.linspace(w0 * 0.25, w0 * 1.75, 100)
             w1_range = np.linspace(w1 * -10, w1 * 10, 100)
         
             # Initialize a zero array to store the sum of least squares (SSE) values for w0, w1, values
             SLS_grid = np.zeros((len(w0_range), len(w1_range)))
         
             # Loop through the range of w0 and w1 values to calculate SSE for each pair
             for i_idx, i in enumerate(w0_range):
                 for j_idx, j in enumerate(w1_range):
                     Y_predicted = j * X + i  # Predicted Y values based on current w0 and w1
                     Sum_of_least_squares = np.sum((Y - Y_predicted)**2)  # SSE for the current w0, w1 pair
                     SLS_grid[i_idx, j_idx] = Sum_of_least_squares # store SSE into a grid
         
             # Find the index of the minimum SSE in the grid
             min_SSE_index = np.unravel_index(np.argmin(SLS_grid), SLS_grid.shape)
             best_w0 = w0_range[min_SSE_index[0]]
             best_w1 = w1_range[min_SSE_index[1]]
         
             # Identify a target SSE to limit our grid search
             target_SSE = SSE_range_mutiplyer * SSE_min
         
             # Find the point in the grid where the SSE is 'SSE_range_mutiplyer' times the minimum SSE
             tolerance = 0.05 * SSE_min  # Allow for small tolerance in SSE
             close_to_target_SSE = np.abs(SLS_grid - target_SSE) < tolerance
         
             # Get the coordinates of the points that are close to the target SSE
             indices = np.where(close_to_target_SSE)
         
             # Extract the corresponding w0 and w1 values
             selected_w0 = w0_range[indices[0]]
             selected_w1 = w1_range[indices[1]]

             # rule out any null w0 and w1 values if they occure
             if selected_w0.size == 0 or selected_w1.size == 0:
                 st.error(f"No valid slope (w1) and intersept (w0) values found that satisfy the SSE condition.")
                 return None, None, None, None, None, None, None, None
                      
             # with in the list of value pairs on the grid with the target SSE value extract the largest and smallest w0 and w1
             min_w0, max_w0 = selected_w0.min(), selected_w0.max()
             min_w1, max_w1 = selected_w1.min(), selected_w1.max()
         
             # Calculate the relationship 'slope' between w0 and w1 (rise over run).
             slope = (max_w1 - min_w1) / (max_w0 - min_w0)
             intercept = min_w1 - slope * min_w0
                  
             return min_w0, max_w0, min_w1, max_w1, slope, intercept, w0, w1
         
         # Call the function to calculate the best-fit line parameters and slope
         min_w0, max_w0, min_w1, max_w1, slope, intersept, w0, w1 = best_fit_line_slope(filtered_df, column_x1, column_y1)
         
         st.write(f"The best-fit line parameters that minimize SSE are:")
         st.write(f"The best-fit line Intersept (w0) = {w0}")   
         st.write(f"Most likly w0 (Intercept) range: {min_w0} to {max_w0}")
         st.write(f"The best-fit line Slpoe (w1) = {w1}")
         st.write(f"Most likly w1 (Slope) range: {min_w1} to {max_w1}")
         st.write(f"The slope and intersept of the surface plot for the reggretion line of variables x and y with endpoints of the surface plot SSE close to 2x minimum SSE is: {slope} {intersept}")

         # create a tool that alows users to adjust the w0, w1 of the regretion along an optimal path determand by SSE grid searching
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
             w0_adjusted = max_w0 - (slider_value / 100.0) * (max_w0 - min_w0) # w0 is inversly related to w1 so change the value of the ad
             w1_adjusted = min_w1 + (slider_value / 100.0) * (max_w1 - min_w1)
             
             # Display the adjusted w0 and w1
             st.write(f"Adjusted w0 (Intercept): {w0_adjusted}")
             st.write(f"Adjusted w1 (Slope): {w1_adjusted}")
         else:
             st.write("Please check the column selections and try again.")
         y_temp = np.log10(filtered_df[column_y1])
         fig6 = px.scatter(filtered_df, x=column_x1, y=y_temp, title=f"PMMoV liner regression model vs {column_x1} {Code2}")
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
         st.write('''
         The two graphs below, fundamentally, the same graph. The difference is what PMMoV is being compared to on the x-axis.
         ''')
         # # Display the plot using Streamlit
         st.plotly_chart(fig6)
         st.plotly_chart(fig7)
         
         st.write('''
                  Based on carful observation of the liner models the best liner regression model involves fitting the data to flow rate and precipitation.
                  The variables used for liner regression were found to correlate with the method of sample collection.
                  Samples collected using a composite collected are most affected by flow rate.
                  The flow rate of a system for composite samples is shown to be negatively correlated to the amount of PMMoV detected on a particular day.
                  Faster than normal moving water through a system is suspected to flush out a system’s fecal mater and thus lower the PMMoV detected.
                  Samples collected directly from the sewer were found to be greatly affected by heavy rainfall.
                  The precipitation recorded over a 24 hour period positively correlates with PMMoV detected on the same day.
                  heavy rainfall is suspected to greatly disturb a sewer environment causing lingering particulates to contaminate the grab sample more than normal, thus increasing the PMMoV detected.
                  Although flow rate and precipitation affect sample collection differently the environmental variable that greatly affects one sample has very little effect on the other sample.
                  The effect each variable has on a sample site is reflected by the weights of the model given.
                  Moving forward, all site log10 PMMoV gc/100ml data is fited to the optimal regression line of flow rate given to a site.
                  While preciption is worth considering using to fit some spific site data, the precipition itself is not suspected of impaecting PMMoV directly and is not well understoded from the data why this relationship occurs.
                  The fited data for each site is stored and pre loaded in the data table as "Log residuals" under the data presentaion tab.
         ''')
         st.write('''
                  Note: For the few wondering why mutivariable liner regression was not used is because the univariable regression is not well understode yet, and we want to avoid overfiting our data.
         ''')





