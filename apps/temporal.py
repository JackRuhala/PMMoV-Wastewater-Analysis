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
