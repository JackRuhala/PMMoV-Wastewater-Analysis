import streamlit as st
from multiapp import MultiApp
from apps import home, basic_intro, extensive_intro, data_cleaning_and_presentaion, environment, regression, lab_preformace # import your app modules here

app = MultiApp()

# Add all your application here
app.add_app("Home", home.app)
app.add_app("Basic Intro", basic_intro.app)
app.add_app("Extensive Intro", extensive_intro.app)
app.add_app("Avalible Data", data_cleaning_and_presentaion.app)
app.add_app("Enviromental Data", environment.app)
app.add_app("Enviromental PMMoV correlation and Regression models", regression.app)
app.add_app("lab Veriables", lab_preformace.app)

# The main app
app.run()
