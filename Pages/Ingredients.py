import streamlit as st
import pandas as pd
from sklearn.datasets import load_wine
import altair as alt

# Load and cache data
@st.cache
def data_load():
    w = load_wine()
    w_df = pd.DataFrame(w.data, columns=w.feature_names)
    w_df["WineType"] = [w.target_names[t] for t in w.target]
    return w_df

w_df = data_load()

# Dashboard layout
st.title("Dashboard for wine Dataset")
st.markdown("This Dashboards shows you Average ingridients in each type of Wine")

# Data summary
if st.checkbox("Data Summary"):
    st.write(w_df.describe())

ingredients = w_df.drop(columns=["WineType"]).columns
avg_w_df = w_df.groupby("WineType").mean().reset_index()

st.subheader("Bar Chart: Average Ingredients per Wine Type")
bar_multiselect = st.multiselect(
    "Select Ingredients", 
    options=ingredients, 
    default=["alcohol", "malic_acid", "ash"],
    help="Select one or more ingredients to visualize their average measurements across different wine types."
)

# Using Altair for custom bar charts
def avg_ing(data, features):
    charts = []
    for feature in features:
        chart = alt.Chart(data).mark_bar().encode(
            x=alt.X('WineType:N', title="Wine Type"),
            y=alt.Y(f'mean({feature}):Q', title="Average Measurement"),
            color='WineType:N',
            tooltip=['WineType:N', f'{feature}:Q']
        ).properties(
            width=700 // len(features), 
            height=300
        ).interactive()
        charts.append(chart)

    combi_chart = alt.hconcat(*charts)
    st.altair_chart(combi_chart, use_container_width=True)

if bar_multiselect:
    avg_ing(avg_w_df, bar_multiselect)
else:
    st.write("Select Atleast an Ingredient to display chart")

