import streamlit as st

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

    st.title('Special Data Source Acknowledgement')
    st.write('''
    This repository focuses on the analysis of viruses extracted from wastewater in Kent County Michigan. 
    While some of the data used for this app was collected from various government agencies (links to the data below), all viral wastewater tracking data was provided by the Grand Valley State University Molecular monitoring team. 
    This repository was given permission by the Grand Valley State lab to make all the data used in this app public. 
    This app will only explore a small aspect of the work being done on wastewater epidemiology so there may be some questions you might have about the data that will only be explained once the Grand Vally team decides to publish their resalts. 
    The conclusions of data made in this app does not speak for the conclusions of the Grand Vally team, although some discussion has taken place to align our thoughts and ideas on the data. 
    I would like to thank the Grand Valley State University Molecular monitoring team for permitting me to use their data and I would encourage others interested in this research to read the publication when it becomes available.
    ''')

    st.write('''
    On the left side menu, go to the ether the basic introduction or the extensive introducton to learn more about the data.
    Click on the data tabs to see spicifc analysis of the data.
    ''')
