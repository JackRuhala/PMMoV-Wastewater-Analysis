import streamlit as st

st.title('Conclusion. What is the Story of the Data')
st.write('''
         Our PMMoV data gives insight into the sewer microenvironment.
         The data tells us that sewer systems all exhibit different and unique properties that make global modeling of sewer data challenging.
         PMMoV was found to be much more resilient to changes in temperature and pH than initially though.
         PMMoV was found to be impacted by environmental flowrate (explains low count outliers) and may exist dormant in the sludge layer of the sewer only to be kicked up when heavy rain is present (explains a Bayesian event that can cause a high outlier).
         PMMoV has a larger than expected variance around the general mean.
         PMMoV variation is likely not caused by known environmental factors, sampling methodology, or laboratory personnel.
         Bi-weekly PMMoV measurements do not suggest any temporal dependents, suggesting possible temporal correlations are weekly or daily if they exist.
         Detection of N1 in a sewer system can predict the deaths due to COVID-19 about a month in advance.
         PMMoV seemingly preforms slightly better than flowrate as a normalizer of N1 lag adjusted residuals, but more data is needed to make any claims.
''')

###-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------###
st.title('Future Proposals for Experiments')
st.write('''
         Now that PMMoV is slightly better understood now that before the analysis of data, more data needs to be collected to fill in the gaps of knowledge.
         One problem found was PMMoV has more variation than expected.
         The unknown variation of the data makes useful predictive modeling methods like oversampling and RBF difficult since we don’t know what target we are supposed to be hitting.
         PMMoV has both a precision problem and an accuracy problem.
         Precision is much easier to explore than accuracy and involves addressing sample distributions on a single day.
         The Prediction experiment involves taking the 500ml lab sample and extracting data from 5 100ml samples instead of 1 100ml sample.
         If 5 samples are taken on a single day, then the variation of sample extractions can be further explored.
         The accuracy problem is much harder to address since it involves having a target.
         The target can be artificially made by collecting more than one human fecal marker and comparing PMMoV to one or multiple human fecal markers.
         Ideally the more human fecal markers you measure, the clearer the true fecal contamination of a sample becomes.
         An established target allows for further refinement of the model for PMMoV and opens more options in terms of making predictions.
         The last experiment explores temporal elements of PMMoV.
         The data concluded that if a temporal element exists, it likely requires more frequent testing to detect.
         An experiment that takes a daily sample of a single site for 2 to 3 weeks could help establish temporal elements on a weekly scale while hourly measurements of a single site can help establish temporal elements on a daily scale.
         Understanding temporal elements will help improve powerful predictive modeling programs like AR(P), and RBF.
         ''')
