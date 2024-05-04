import streamlit as st
from sklearn.datasets import load_wine
import pandas as pd
import altair as alt

@st.cache
def load_data():
    file = load_wine()
    df = pd.DataFrame(file.data, columns=file.feature_names)
    df['wine_type'] = [file.target_names[i] for i in file.target]
    return df

# Load the data
df = load_data()

def hist(data, column):
    chart = alt.Chart(data).mark_bar().encode(
        alt.X(column, bin=True),
        y='count()',
        tooltip=[column, 'count()']
    ).properties(width=600, height=400)
    st.altair_chart(chart)

def scatter(data, x, y):
    chart = alt.Chart(data).mark_circle(size=60).encode(
        x=x,
        y=y,
        color='wine_type',
        tooltip=[x, y, 'wine_type']
    ).interactive().properties(
        width=600,
        height=400
    )
    st.altair_chart(chart)

def pie(data, column):
    chart = alt.Chart(data).mark_arc(innerRadius=0).encode(
        theta=alt.Theta(f"{column}:Q", stack=True),
        color=alt.Color(f"{column}:N", legend=None),
        tooltip=[f"{column}:N", f"{column}:Q"]
    ).properties(width=400, height=400)
    st.altair_chart(chart)

def donut(data, column):
    chart = alt.Chart(data).mark_arc(innerRadius=100).encode(
        theta=alt.Theta(f"{column}:Q", stack=True),
        color=alt.Color(f"{column}:N", legend=None),
        tooltip=[f"{column}:N", f"{column}:Q"]
    ).properties(width=400, height=400)
    st.altair_chart(chart)

def combi(data, bar_col, line_col):
    base = alt.Chart(data)
    
    bar = base.mark_bar(opacity=0.7, color='blue').encode(
        x=alt.X(bar_col, axis=alt.Axis(title=bar_col)),
        y=alt.Y('count()', axis=alt.Axis(title='Count')),
        tooltip=[bar_col, 'count()']
    )

    line = base.mark_line(color='red').encode(
        x=alt.X(bar_col, axis=alt.Axis(title=bar_col)),
        y=alt.Y(f'mean({line_col})', axis=alt.Axis(title=f'Mean of {line_col}', titleColor='red')),
        tooltip=[bar_col, f'mean({line_col})']
    )

    chart = alt.layer(bar, line).resolve_scale(
        y='independent'
    ).properties(width=600, height=400)

    st.altair_chart(chart)

# Sidebar for user input
st.sidebar.header('Wine Dataset Visualization')
chart_type = st.sidebar.selectbox(
    'Select Chart Type',
    ['Histogram', 'Scatter Plot', 'Pie Chart', 'Donut Chart', 'Combination Chart']
)

if chart_type == 'Histogram':
    column = st.sidebar.selectbox('Select column for histogram', df.columns[:-1])
    hist(df, column)
elif chart_type == 'Scatter Plot':
    x_axis = st.sidebar.selectbox('Select x axis', df.columns, index=0)
    y_axis = st.sidebar.selectbox('Select y axis', df.columns, index=1)
    scatter(df, x_axis, y_axis)
elif chart_type in ['Pie Chart', 'Donut Chart']:
    column = st.sidebar.selectbox('Select column', df.columns)
    if chart_type == 'Pie Chart':
        pie(df, column)
    elif chart_type == 'Donut Chart':
        donut(df, column)
elif chart_type == 'Combination Chart':
    bar_col = st.sidebar.selectbox('Select column for Bar', df.columns[:-1])
    line_col = st.sidebar.selectbox('Select column for Line', df.columns)
    combi(df, bar_col, line_col)
