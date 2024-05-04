import streamlit as st
from sklearn.datasets import load_wine
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Load the wine dataset and cache it to avoid reloading on every interaction.
@st.cache(allow_output_mutation=True)
def load_data():
    w = load_wine()
    w_df = pd.DataFrame(w.data, columns=w.feature_names)
    w_df["Winetype"] = [w.target_names[t] for t in w.target]
    return w_df

# Load data
w_df = load_data()

# Dropdown to select features for plotting
st.sidebar.header("Visualization Settings")
f_s = st.sidebar.multiselect('Select features to plot:', w_df.columns[:-1], default=w_df.columns[0])

# Plotting
if f_s:
    st.header("Feature Distribution by Wine Type")
    for feature in f_s:
        fig, ax = plt.subplots()
        sns.boxplot(x='Winetype', y=feature, data=w_df)
        plt.title(f'Distribution of {feature} by Wine Type')
        st.pyplot(fig)

# Display the raw data
st.header("Wine Dataset Overview")
st.dataframe(w_df)

# Group by 'Winetype' and calculate mean for each numerical feature
avg_w_df = w_df.groupby("Winetype").mean().reset_index()

st.header("Average Values by Wine Type")
st.dataframe(avg_w_df)
