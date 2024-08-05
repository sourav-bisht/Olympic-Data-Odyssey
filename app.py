import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
import helper
import preprocessor


# Caching the data loading process

@st.cache_data
def load_data():
    df = pd.read_csv('athlete_events.csv')
    region_df = pd.read_csv('noc_regions.csv')
    df = preprocessor.preprocess(df, region_df)
    return df


df = load_data()

st.sidebar.title("Olympics Analysis")
user_menu = st.sidebar.radio(
    "Select an analysis type:",
    ('Medal Tally', 'Overall Analysis', 'Country-Wise Analysis', 'Athlete-Wise Analysis')
)

if user_menu == 'Medal Tally':
    st.sidebar.header("Medal Tally")
    years, country = helper.country_year_list(df)

    selected_year = st.sidebar.selectbox("Select Year", years)
    selected_country = st.sidebar.selectbox("Select Country", country)

    medal_tally = helper.fetch_medal_tally(df, selected_year, selected_country)
    if selected_country == 'Overall' and selected_year == 'Overall':
        st.title("Overall Tally")
    if selected_year != 'Overall' and selected_country == 'Overall':
        st.title("Medal Tally in " + str(selected_year) + " Olympics")
    if selected_year == 'Overall' and selected_country != 'Overall':
        st.title(selected_country + " Overall Performance")
    if selected_year != 'Overall' and selected_country != 'Overall':
        st.title("Performance of " + selected_country + " in Year " + str(selected_year))

    st.table(medal_tally)

if user_menu == 'Overall Analysis':
    editions = df['Year'].nunique() - 1
    cities = df['City'].nunique()
    sports = df['Sport'].nunique()
    events = df['Event'].nunique()
    athletes = df['Name'].nunique()
    nations = df['region'].nunique()

    st.title("Top Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.header("Editions")
        st.title(editions)
    with col2:
        st.header("Hosts")
        st.title(cities)
    with col3:
        st.header("Sports")
        st.title(sports)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.header("Events")
        st.title(events)
    with col2:
        st.header("Nations")
        st.title(nations)
    with col3:
        st.header("Athletes")
        st.title(athletes)

    nations_over_time = helper.data_over_time(df, 'region')
    fig = px.line(nations_over_time, x="Edition", y="region", labels={"region": "No. of Region"})
    st.title("Participating Nations over the years")
    st.plotly_chart(fig)

    events_over_time = helper.data_over_time(df, 'Event')
    fig = px.line(events_over_time, x="Edition", y="Event", labels={"Event": "No. of Events"})
    st.title("Events over the years")
    st.plotly_chart(fig)

    athletes_over_time = helper.data_over_time(df, 'Name')
    fig = px.line(athletes_over_time, x="Edition", y="Name", labels={"Name": "No. of Athletes"})
    st.title("Athletes participation over the years")
    st.plotly_chart(fig)

    st.title("No. of Events over the  years")
    fig, ax = plt.subplots(figsize=(25, 25))
    x = df.drop_duplicates(['Year', 'Sport', 'Event'])
    ax = sns.heatmap(x.pivot_table(index='Sport', columns='Year', values='Event', aggfunc='count').fillna(0).astype(int)
                     , annot=True)
    st.pyplot(fig)

    st.title("Most Successful Athletes")
    sport_list = df['Sport'].unique().tolist()
    sport_list.sort()
    sport_list.insert(0, 'Overall')

    selected_sport = st.selectbox('Select a Sport', sport_list)

    x = helper.most_successful(df, selected_sport)
    st.table(x)

if user_menu == 'Country-Wise Analysis':
    st.sidebar.title('Country-Wise Analysis')
    country_list = df['region'].dropna().unique().tolist()
    country_list.sort()

    selected_country = st.sidebar.selectbox('Select a Country', country_list)

    country_df = helper.year_wise_medal_tally(df, selected_country)
    fig = px.line(country_df, x="Year", y="Medal")
    st.title(selected_country + " Medal Tally over the years")
    st.plotly_chart(fig)

    st.title(selected_country + " Excels in the following sports")
    pt = helper.country_event_heatmap(df, selected_country)
    if pt.empty:
        st.warning("No data available of "+selected_country+" for the heatmap.")
    else:
        pt = pt.fillna(0)
        try:
            fig, ax = plt.subplots(figsize=(25, 25))
            ax = sns.heatmap(pt, annot=True)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"An error occurred while plotting the heatmap: {e}")

    st.title("Top 10 athletes of " + selected_country)
    top10_df = helper.most_successful_country_wise(df, selected_country)
    st.table(top10_df)

if user_menu == 'Athlete-Wise Analysis':
    athlete_df = df.drop_duplicates(subset=['Name', 'region'])
    x1 = athlete_df['Age'].dropna()
    x2 = athlete_df[athlete_df['Medal'] == 'Gold']['Age'].dropna()
    x3 = athlete_df[athlete_df['Medal'] == 'Silver']['Age'].dropna()
    x4 = athlete_df[athlete_df['Medal'] == 'Bronze']['Age'].dropna()

    fig = go.Figure()

    for data, name in zip([x1, x2, x3, x4], ['Overall Age', 'Gold Medalist', 'Silver Medalist', 'Bronze Medalist']):
        kde = np.histogram(data, bins=30, density=True)
        x = (kde[1][:-1] + kde[1][1:]) / 2
        y = kde[0]
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines',
            name=name
        ))
    fig.update_layout(autosize=False, width=1000, height=700,
                      xaxis_title='Age',
                      yaxis_title='Probability Density')
    st.title("Distribution of Age")
    st.plotly_chart(fig)

    famous_sports = ['Basketball', 'Judo', 'Football', 'Tug-Of-War', 'Athletics',
                     'Swimming', 'Badminton', 'Sailing', 'Gymnastics',
                     'Art Competitions', 'Handball', 'Weightlifting', 'Wrestling',
                     'Water Polo', 'Hockey', 'Rowing', 'Fencing',
                     'Shooting', 'Boxing', 'Taekwondo', 'Cycling', 'Diving', 'Canoeing',
                     'Tennis', 'Golf', 'Softball', 'Archery',
                     'Volleyball', 'Synchronized Swimming', 'Table Tennis', 'Baseball',
                     'Rhythmic Gymnastics', 'Rugby Sevens',
                     'Beach Volleyball', 'Triathlon', 'Rugby', 'Polo', 'Ice Hockey']
    x = []
    names = []
    for sport in famous_sports:
        temp_df = athlete_df[athlete_df['Sport'] == sport]
        ages = temp_df[temp_df['Medal'] == 'Gold']['Age'].dropna()
        if not ages.empty:
            x.append(ages)
            names.append(sport)

    fig = go.Figure()

    for data, name in zip(x, names):
        kde = gaussian_kde(data, bw_method='scott')  # Adjust bandwidth method
        x_range = np.linspace(min(data), max(data), 1000)  # Increase the number of points
        y = kde(x_range)

        fig.add_trace(go.Scatter(
            x=x_range,
            y=y,
            mode='lines',
            name=name
        ))
    fig.update_layout(
        title='Age Distribution of Gold Medalists by Sport',
        xaxis_title='Age',
        yaxis_title='Probability Density',
        autosize=False,
        width=1000,
        height=700
    )

    st.title("Distribution of Age w.r.t Sports")
    st.plotly_chart(fig)

    sport_list = df['Sport'].unique().tolist()
    sport_list.sort()
    sport_list.insert(0, 'Overall')

    st.title('Height Vs Weight')
    selected_sport = st.selectbox('Select a Sport', sport_list)
    temp_df = helper.weight_v_height(df, selected_sport)
    fig, ax = plt.subplots()
    ax = sns.scatterplot(x=temp_df['Weight'], y=temp_df['Height'], hue=temp_df['Medal'], style=temp_df['Sex'], s=50)
    st.pyplot(fig)

    st.title("Men Vs Women over the years")
    final = helper.men_vs_women(df)
    fig = px.line(final, x='Year', y=['Male', 'Female'])
    fig.update_layout(autosize=False, width=1000, height=700)
    st.plotly_chart(fig)


st.sidebar.markdown("---")  
st.sidebar.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 100%;
    }
    .sidebar .footer {
        font-size: 14px;
        color: grey;
        text-align: center;
        padding-top: 10px;
    }
    </style>
    <div class="footer">
        Made by :- Sourav Bisht
        <br>
        Graphic Era Hill University, Dehradun
    </div>
    """, unsafe_allow_html=True
)
