import streamlit as st
from multiapp import MultiApp
from apps import home, data_stats # import your app modules here

app = MultiApp()

# Add all your application here
app.add_app("Home", Home.app)
app.add_app("Basic Intro", Basic Intro.app)
app.add_app("Extensive Intro", Extensive intro.app)

# The main app
app.run()
