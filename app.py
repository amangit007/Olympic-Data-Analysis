import streamlit as st
import pandas as pd
import preprocessor,helper
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff

df = pd.read_csv('athlete_events.csv')
region_df = pd.read_csv('noc_regions.csv')



st.sidebar.title('Olympic Analysis')
a=['Summer','Winter','Combined']
selected= st.sidebar.selectbox('Select Type',a)
df = preprocessor.preprocess(df, region_df,selected)

user_menu = st.sidebar.radio(
    'Select an Option',
    ('medal Tally', 'overall', 'country wise', 'Athlete wise')
)

if user_menu == 'medal Tally':
    st.sidebar.header('Medal Tally')
    years,country=helper.country_year_list(df)

    selected_year=st.sidebar.selectbox('select year',years)
    selected_country=st.sidebar.selectbox('select country',country)

    medal_tall = helper.medal_tally(df,selected_year,selected_country)
    if selected_year == 'Overall' and selected_country == 'Overall':
        st.title("Overall Tally")
    if selected_year != 'Overall' and selected_country == 'Overall':
        st.title("Medal Tally in " + str(selected_year) + " Olympics")
    if selected_year == 'Overall' and selected_country != 'Overall':
        st.title(selected_country + " overall performance")
    if selected_year != 'Overall' and selected_country != 'Overall':
        st.title(selected_country + "'s performance in " + str(selected_year) + " Olympics")
    st.table(medal_tall)

if user_menu== 'overall':
    editions = df['Year'].unique().shape[0] - 1
    cities = df['City'].unique().shape[0]
    sports = df['Sport'].unique().shape[0]
    events = df['Event'].unique().shape[0]
    athletes = df['Name'].unique().shape[0]
    nations = df['region'].unique().shape[0]

    st.title("Top Statistics")
    col1, col2, col3 = st.beta_columns(3)
    with col1 :
        st.header("Editions")
        st.title(editions)
    with col2 :
        st.header("Hosts")
        st.title(cities)
    with col3 :
        st.header("Sports")
        st.title(sports)

    col1, col2, col3 = st.beta_columns(3)
    with col1 :
        st.header("Events")
        st.title(events)
    with col2 :
        st.header("Nations")
        st.title(nations)
    with col3 :
        st.header("Athletes")
        st.title(athletes)

    nations_over_time = helper.data_over_time(df, 'region')
    fig = px.line(nations_over_time, x="Edition", y="region")
    st.title("Participating Nations over the years")
    st.plotly_chart(fig)

    events_over_time = helper.data_over_time(df, 'Event')
    fig = px.line(events_over_time, x="Edition", y="Event")
    st.title("Events over the years")
    st.plotly_chart(fig)

    athlete_over_time = helper.data_over_time(df, 'Name')
    fig = px.line(athlete_over_time, x="Edition", y="Name")
    st.title("Athletes over the years")
    st.plotly_chart(fig)


    st.title("Most successful Athletes")
    sport_list = df['Sport'].unique().tolist()
    sport_list.sort()
    sport_list.insert(0, 'Overall')

    selected_sport = st.selectbox('Select a Sport', sport_list)
    x = helper.most_successful(df, selected_sport)
    st.table(x)

    st.title("No. of Events over time(Every Sport)")
    fig ,ax= plt.subplots(figsize=(20, 20))
    x = df.drop_duplicates(['Year', 'Sport', 'Event'])
    ax = sns.heatmap(
    x.pivot_table(index='Sport', columns='Year', values='Event', aggfunc='count').fillna(0).astype('int'),
    annot=True)
    st.pyplot(fig)

if user_menu=='country wise':
    st.sidebar.title("country wise Analysis")
    country_df=df['region'].dropna().unique().tolist()
    country_df.sort()
    country_select=st.sidebar.selectbox("Select the Country",country_df)

    final_df=helper.yearwise_medal_tally(df,country_select)
    fig=px.line(final_df,x='Year',y='Medal')
    st.title(country_select + "'s Medal Tally over the years")
    st.plotly_chart(fig)

    f = ['Albania', 'American Samoa', 'Andorra', 'Angola', 'Antigua', 'Aruba', 'Bangladesh',
         'Belize', 'Benin', 'Bhutan', 'Boliva', 'Bosnia and Herzegovina', 'Brunei', 'Burkina Faso',
         'Cambodia', 'Cape Verde', 'Cayman Islands', 'Central African Republic', 'Chad', 'Comoros',
         'Cook Islands', 'Democratic Republic of the Congo', 'Dominica', 'El Salvador',
         'Equatorial Guinea', 'Gambia', 'Guam', 'Guinea', 'Guinea-Bissau', 'Honduras', 'Kiribati',
         'Laos', 'Lesotho', 'Liberia', 'Libya', 'Madagascar', 'Malawi', 'Maldives', 'Mali', 'Malta',
         'Marshall Islands', 'Mauritania', 'Micronesia', 'Myanmar', 'Nauru', 'Nicaragua', 'Oman',
         'Palau', 'Palestine', 'Papua New Guinea', 'Republic of Congo', 'Rwanda', 'Saint Kitts',
         'Saint Lucia', 'Saint Vincent', 'Samoa', 'San Marino', 'Sao Tome and Principe', 'Seychelles',
         'Sierra Leone', 'Solomon Islands', 'Somalia', 'South Sudan', 'Swaziland', 'Timor-Leste',
         'Turkmenistan', 'Vanuatu', 'Virgin Islands, British', 'Yemen']

    st.title(country_select + " excels in the following sports")
    if country_select in f:
        st.info('Country selected have never won any medal')
    else:
        pt = helper.country_event_heatmap(df, country_select)
        fig, ax = plt.subplots(figsize=(15, 15))
        ax = sns.heatmap(pt, annot=True)
        st.pyplot(fig)

        st.title("Top 10 athletes of " + country_select)
        top10_df = helper.most_successful_countrywise(df, country_select)
        st.table(top10_df)


if user_menu == 'Athlete wise':
    athlete_df = df.drop_duplicates(subset=['Name', 'region'])

    x1 = athlete_df['Age'].dropna()
    x2 = athlete_df[athlete_df['Medal'] == 'Gold']['Age'].dropna()
    x3 = athlete_df[athlete_df['Medal'] == 'Silver']['Age'].dropna()
    x4 = athlete_df[athlete_df['Medal'] == 'Bronze']['Age'].dropna()

    fig = ff.create_distplot([x1, x2, x3, x4], ['Overall Age', 'Gold Medalist', 'Silver Medalist', 'Bronze Medalist'],show_hist=False, show_rug=False)
    fig.update_layout(autosize=False,width=1000,height=600)
    st.title("Distribution of Age")
    st.plotly_chart(fig)


    if selected == 'Summer':
        x = []
        name = []
        famous_sports = ['Basketball', 'Judo', 'Football', 'Tug-Of-War', 'Athletics',
                         'Swimming', 'Badminton', 'Sailing', 'Gymnastics',
                         'Art Competitions', 'Handball', 'Weightlifting', 'Wrestling',
                         'Water Polo', 'Hockey', 'Rowing', 'Fencing',
                         'Shooting', 'Boxing', 'Taekwondo', 'Cycling', 'Diving', 'Canoeing',
                         'Tennis', 'Golf', 'Softball', 'Archery',
                         'Volleyball', 'Synchronized Swimming', 'Table Tennis', 'Baseball',
                         'Rhythmic Gymnastics', 'Rugby Sevens',
                         'Beach Volleyball', 'Triathlon', 'Rugby', 'Polo', 'Ice Hockey']
        for sport in famous_sports :
            temp_df = athlete_df[athlete_df['Sport'] == sport]
            x.append(temp_df[temp_df['Medal'] == 'Gold']['Age'].dropna())
            name.append(sport)

        fig = ff.create_distplot(x, name, show_hist=False, show_rug=False)
        fig.update_layout(autosize=False, width=1000, height=600)
        st.title("Distribution of Age wrt Sports(Gold Medalist)")
        st.plotly_chart(fig)

    if selected == 'Winter' :
        x = []
        name = []
        famous_sports = ['Speed Skating', 'Cross Country Skiing', 'Ice Hockey', 'Biathlon',
       'Alpine Skiing', 'Luge', 'Bobsleigh', 'Figure Skating',
       'Nordic Combined', 'Freestyle Skiing', 'Ski Jumping', 'Curling',
       'Snowboarding', 'Short Track Speed Skating', 'Skeleton',
       'Military Ski Patrol', 'Alpinism']
        for sport in famous_sports :
            temp_df = athlete_df[athlete_df['Sport'] == sport]
            x.append(temp_df[temp_df['Medal'] == 'Gold']['Age'].dropna())
            name.append(sport)

        fig = ff.create_distplot(x, name, show_hist=False, show_rug=False)
        fig.update_layout(autosize=False, width=1000, height=600)
        st.title("Distribution of Age wrt Sports(Gold Medalist)")
        st.plotly_chart(fig)

    if selected == 'Combined' :
        x = []
        name = []
        famous_sports = ['Basketball', 'Judo', 'Football', 'Tug-Of-War', 'Speed Skating',
       'Cross Country Skiing', 'Athletics', 'Ice Hockey', 'Swimming',
       'Badminton', 'Sailing', 'Biathlon', 'Gymnastics',
       'Art Competitions', 'Alpine Skiing', 'Handball', 'Weightlifting',
       'Wrestling', 'Luge', 'Water Polo', 'Hockey', 'Rowing', 'Bobsleigh',
       'Fencing', 'Equestrianism', 'Shooting', 'Boxing', 'Taekwondo',
       'Cycling', 'Diving', 'Canoeing', 'Tennis', 'Modern Pentathlon',
       'Figure Skating', 'Golf', 'Softball', 'Archery', 'Volleyball',
       'Synchronized Swimming', 'Table Tennis', 'Nordic Combined',
       'Baseball', 'Rhythmic Gymnastics', 'Freestyle Skiing',
       'Rugby Sevens', 'Trampolining', 'Beach Volleyball', 'Triathlon',
       'Ski Jumping', 'Curling', 'Snowboarding', 'Rugby',
       'Short Track Speed Skating', 'Skeleton', 'Lacrosse', 'Polo',
       'Cricket', 'Racquets', 'Motorboating', 'Military Ski Patrol',
       'Croquet', 'Jeu De Paume', 'Roque', 'Alpinism', 'Basque Pelota',
       'Aeronautics']
        for sport in famous_sports :
            temp_df = athlete_df[athlete_df['Sport'] == sport]
            x.append(temp_df[temp_df['Medal'] == 'Gold']['Age'].dropna())
            name.append(sport)

        fig = ff.create_distplot(x, name, show_hist=False, show_rug=False)
        fig.update_layout(autosize=False, width=1000, height=600)
        st.title("Distribution of Age wrt Sports(Gold Medalist)")
        st.plotly_chart(fig)


    sport_list = df['Sport'].unique().tolist()
    sport_list.sort()
    sport_list.insert(0, 'Overall')

    st.title('Height Vs Weight')
    selected_sport = st.selectbox('Select a Sport', sport_list)
    temp_df = helper.weight_v_height(df, selected_sport)
    fig, ax = plt.subplots()
    ax = sns.scatterplot(temp_df['Weight'], temp_df['Height'], hue=temp_df['Medal'], style=temp_df['Sex'], s=60)
    st.pyplot(fig)

    st.title("Men Vs Women Participation Over the Years")
    final = helper.men_vs_women(df)
    fig = px.line(final, x="Year", y=["Male", "Female"])
    fig.update_layout(autosize=False, width=1000, height=600)
    st.plotly_chart(fig)



