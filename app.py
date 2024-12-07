import streamlit as st
from multiapp import MultiApp
from apps import home, basic_intro, extensive_intro # import your app modules here

app = MultiApp()

# Add all your application here
app.add_app("Home", home.app)
app.add_app("Basic Intro", basic_intro.app)
app.add_app("Extensive Intro", extensive_intro.app)

# The main app
app.run()
