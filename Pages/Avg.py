import streamlit as st
from sklearn.datasets import load_wine
import pandas as pd

@st.cache_data
def data_load(): ## Dataset
    w = load_wine()
    w_df = pd.DataFrame(w.data, columns=w.feature_names) 
    w_df["WineType"] = [w.target_names[t] for t in w.target]
    return w_df

w_df = data_load()
ingredients = w_df.drop(columns=["WineType"]).columns

avg_wine_df = w_df.groupby("WineType").mean().reset_index() ## Avg Ingredients.

st.header("Bar Chart: Average Ingredients per Wine Type.")

bar_multiselect = st.multiselect(label="Ingredients", options=ingredients, default=["alcohol", "malic_acid", "ash"])

st.bar_chart(avg_wine_df, x="WineType", y=bar_multiselect, height=500, use_container_width=True)