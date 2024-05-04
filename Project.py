import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine

wine_dataset = load_wine()

st.title(":Red[Wine Dataset] :green[Analysis] :tea: :coffee: :chart: :bar_chart:")
st.markdown("Exploring the correlation between the componenets used in crafting different categories of Wines")

st.write(wine_dataset.DESCR)


     
