import streamlit as st
import pandas as pd
import initial, functions
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import numpy as np
from PIL import Image

df = pd.read_csv('athlete_events.csv')
region_df = pd.read_csv('noc_regions.csv')

st.sidebar.title('Olympic Analysis')
a = ['Summer', 'Winter', 'Combined']
selected = st.sidebar.selectbox('Select Type', a)
df = initial.type_select(df, region_df, selected)

user_menu = st.sidebar.radio(
    'Select an Option',
    ('Olympics', 'Medal Tally', 'Year wise', 'Country wise', 'Athlete wise')
)

if user_menu == 'Medal Tally' :
    st.image('static/medal.jpg')
    st.sidebar.header('Medal Tally')
    years, country = functions.country_year_list(df)

    selected_year = st.sidebar.selectbox('Select year', years)
    selected_country = st.sidebar.selectbox('Select country', country)

    medal_tall = functions.medal_tally(df, selected_year, selected_country)
    medal_tall.index = medal_tall.index + 1
    if selected_year == 'Overall' and selected_country == 'Overall' :
        st.title("Overall Tally")
    if selected_year != 'Overall' and selected_country == 'Overall' :
        st.title("Medal Tally in " + str(selected_year) + " Olympics")
    if selected_year == 'Overall' and selected_country != 'Overall' :
        st.title(selected_country + " overall performance")
    if selected_year != 'Overall' and selected_country != 'Overall' :
        st.title(selected_country + "'s performance in " + str(selected_year) + " Olympics")
    medal_tall.index = np.arange(1, len(medal_tall) + 1)
    st.table(medal_tall)

if user_menu == 'Olympics' :
    st.image('static/olympic.jpg')
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

    nations_over_time = functions.data_over_time(df, 'region')
    fig = px.line(nations_over_time, x="Edition", y="region")
    st.title("Participating Nations over the years")
    st.plotly_chart(fig)

    events_over_time = functions.data_over_time(df, 'Event')
    fig = px.line(events_over_time, x="Edition", y="Event")
    st.title("Events over the years")
    st.plotly_chart(fig)

    athlete_over_time = functions.data_over_time(df, 'Name')
    fig = px.line(athlete_over_time, x="Edition", y="Name")
    st.title("Athletes over the years")
    st.plotly_chart(fig)

    st.title("Most successful Athletes")
    sport_list = df['Sport'].unique().tolist()
    sport_list.sort()
    sport_list.insert(0, 'Overall')

    selected_sport = st.selectbox('Select a Sport', sport_list)
    x = functions.most_successful(df, selected_sport)
    x.index = np.arange(1, len(x) + 1)
    st.table(x)

    st.title("No. of Events over time(Every Sport)")
    fig, ax = plt.subplots(figsize=(20, 20))
    x = df.drop_duplicates(['Year', 'Sport', 'Event'])
    ax = sns.heatmap(
        x.pivot_table(index='Sport', columns='Year', values='Event', aggfunc='count').fillna(0).astype('int'),
        annot=True)
    st.pyplot(fig)

    athlete_df = df.drop_duplicates(subset=['Name', 'region'])

    x1 = athlete_df['Age'].dropna()
    x2 = athlete_df[athlete_df['Medal'] == 'Gold']['Age'].dropna()
    x3 = athlete_df[athlete_df['Medal'] == 'Silver']['Age'].dropna()
    x4 = athlete_df[athlete_df['Medal'] == 'Bronze']['Age'].dropna()

    fig = ff.create_distplot([x1, x2, x3, x4], ['Overall Age', 'Gold Medalist', 'Silver Medalist', 'Bronze Medalist'],
                             show_hist=False, show_rug=False)
    fig.update_layout(autosize=False, width=1000, height=600)
    st.title("Distribution of Age")
    st.plotly_chart(fig)

    if selected == 'Summer' :
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

    sport_list13 = df['Sport'].unique().tolist()
    sport_list13.sort()
    sport_list13.insert(0, 'Overall')

    st.title("Men Vs Women Participation Over the Years")
    final = functions.men_vs_women(df)
    fig = px.line(final, x="Year", y=["Male", "Female"])
    fig.update_layout(autosize=False, width=1000, height=600)
    st.plotly_chart(fig)

    st.title('Height Vs Weight')
    selected_sport12 = st.selectbox('Select a Sport : ', sport_list13)
    temp_df = functions.weight_v_height(df, selected_sport12)
    fig, ax = plt.subplots()
    ax = sns.scatterplot(temp_df['Weight'], temp_df['Height'], hue=temp_df['Medal'], style=temp_df['Sex'], s=60)
    st.pyplot(fig)

if user_menu == 'Country wise' :
    st.sidebar.title("Country wise Analysis")
    country_df = df['region'].dropna().unique().tolist()
    country_df.sort()
    country_select = st.sidebar.selectbox("Select the Country", country_df)

    final_df = functions.yearwise_medal_tally(df, country_select)
    fig = px.line(final_df, x='Year', y='Medal')
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

    st.title(country_select + "'s participation in Olympics")

    pt = functions.country_event_heatmap(df, country_select, )
    fig, ax = plt.subplots(figsize=(15, 15))
    ax = sns.heatmap(pt, annot=True)
    st.pyplot(fig)

    st.title(country_select + "'s success in Olympics(Medals)")
    if country_select in f :
        st.info('Country selected have never won any medal')
    else :
        pt = functions.country_event_heatmap2(df, country_select)
        fig, ax = plt.subplots(figsize=(15, 15))
        ax = sns.heatmap(pt, annot=True)
        st.pyplot(fig)

        st.title("Top 10 athletes of " + country_select)
        top10_df = functions.most_successful_countrywise(df, country_select)
        top10_df.index = np.arange(1, len(top10_df) + 1)
        st.table(top10_df)

if user_menu == 'Athlete wise' :
    country_select = df['region'].dropna().unique().tolist()
    country_select.sort()
    st.sidebar.title('Select country')
    country_selecto = st.sidebar.selectbox('Select Country', country_select)
    new_df = df[df['region'] == country_selecto]

    name_list = new_df['Name'].unique().tolist()
    name_list.sort()

    st.sidebar.title('Select Athlete')
    name_select = st.sidebar.selectbox('Select Athlete', name_list)

    st.title(name_select)
    stand_df = df[df['Name'] == name_select]

    stand_df2 = stand_df['Sex'].dropna().unique().tolist()
    stand_df2 = stand_df2[0]
    stand_df3 = stand_df['Season'].dropna().unique().tolist()
    stand_df4 = stand_df['Sport'].dropna().unique().tolist()

    if stand_df2 == 'M' :
        stand_df2 = 'Male'

    if stand_df2 == 'F' :
        stand_df2 = 'Female'

    str1 = ''
    for i in stand_df3 :
        str1 = str1 + i + ' '

    str2 = ''
    for j in stand_df4 :
        str2 = str2 + j + ' '

    col1, col2, col3, col4 = st.beta_columns(4)
    with col1 :
        st.header("Country")
        st.title(country_selecto)
    with col2 :
        st.header("Gender")
        st.title(stand_df2)
    with col3 :
        st.header("Type")
        st.title(str1)
    with col4 :
        st.header("Sport")
        st.title(str2)

    stand_df5 = stand_df.sum()[['Gold', 'Silver', 'Bronze']]
    goldis = stand_df5['Gold']
    silvis = stand_df5['Silver']
    bronis = stand_df5['Bronze']
    totis = stand_df5['Gold'] + stand_df5['Silver'] + stand_df5['Bronze']
    st.title(' ')
    st.title('All Time Medals')
    col1, col2, col3, col4 = st.beta_columns(4)
    with col1 :
        st.header("Gold")
        st.title(goldis)
    with col2 :
        st.header("Silver")
        st.title(silvis)
    with col3 :
        st.header("Bronze")
        st.title(bronis)
    with col4 :
        st.header("Total")
        st.title(totis)

    st.title(' ')
    st.title('Performance accross the years')

    stand_df = stand_df[['Games', 'City', 'Age', 'Height', 'Weight', 'Team', 'Event', 'Medal']]
    stand_df = stand_df.fillna({'Age' : 'No Data', 'Height' : 'No Data', 'Weight' : 'No Data', 'Medal' : 'No Medal'})
    stand_df.index = np.arange(1, len(stand_df) + 1)
    st.table(stand_df)

if user_menu == 'Year wise' :
    game_select = df['Games'].dropna().unique().tolist()
    game_select.sort()
    game_selected = st.sidebar.selectbox('Select the games', game_select)

    game_df = df[df['Games'] == game_selected]
    sports = game_df['Sport'].unique().shape[0]
    events = game_df['Event'].unique().shape[0]
    athletes = game_df['Name'].unique().shape[0]
    nations = game_df['region'].unique().shape[0]

    city_name = game_df['City'].unique().tolist()[0]

    get_path = game_selected.split(" ")
    new_path = get_path[0] + "_" + get_path[1]

    image = Image.open('static/' + new_path + ".jpg")
    col1, col2 = st.beta_columns(2)
    with col1 :
        st.image(image)
    with col2 :
        st.title(get_path[0])
        st.title(city_name + ' Olympics')

    col1, col2, col3, col4 = st.beta_columns(4)
    with col1 :
        st.header('Sports')
        st.title(sports)
    with col2 :
        st.header('Events')
        st.title(events)
    with col3 :
        st.header('Nations')
        st.title(nations)
    with col4 :
        st.header('Athletes')
        st.title(athletes)

    st.title('Successfull Athletes')
    sport_select = game_df['Sport'].dropna().unique().tolist()
    sport_select.sort()
    sport_select.insert(0, 'Overall')
    selected = st.selectbox('Select Sport', sport_select)

    event_df = game_df[game_df['Sport'] == selected]
    event_df = event_df['Event'].dropna().unique().tolist()
    event_df.sort()
    event_df.insert(0, 'Overall')
    evenselecto = st.selectbox('Select Event', event_df)

    finished_df = functions.success(game_df, selected, evenselecto)
    finished_df.index = np.arange(1, len(finished_df) + 1)
    st.table(finished_df)

    st.title('Atheletes by country')
    ath_df = game_df.drop_duplicates(subset=['Name'])
    ath_df = ath_df.groupby('Team').count()['Name'].reset_index().sort_values(['Name'], ascending=False)
    ath_df = ath_df.rename(columns={'region' : 'Country', 'Name' : 'Athletes'})
    ath_df.index = np.arange(1, len(ath_df) + 1)

    st.table(ath_df)

st.sidebar.text('* To view properly change Appearance to wide mode ')
