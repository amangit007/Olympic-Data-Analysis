import numpy as np
import pandas as pd

def medal_tally(df,year,country):
    medal_df = df.drop_duplicates(subset=['Team', 'NOC', 'Games', 'Year', 'City', 'Sport', 'Event', 'Medal'])
    flame=0
    if (year=='Overall') and (country=='Overall'):
        x=medal_df
    if (year=='Overall') and (country!='Overall'):
        flame=1
        x=medal_df[medal_df['region']==country]
    if (year != 'Overall') and (country == 'Overall') :
        x = medal_df[medal_df['Year'] == year]
    if (year != 'Overall') and (country != 'Overall') :
        x  = medal_df[(medal_df['Year'] == year) & (medal_df['region'] == country)]

    if flame==1:
        medal_df=x.groupby('Year').sum()[['Gold', 'Silver', 'Bronze']].sort_values('Year').reset_index()
    else:
        medal_df = x.groupby('region').sum()[['Gold', 'Silver', 'Bronze']].sort_values(['Gold', 'Silver', 'Bronze'], ascending=False).reset_index()

    medal_df['Total']=medal_df['Gold']+ medal_df['Silver'] +medal_df['Bronze']

    medal_df['Gold'] = medal_df['Gold'].astype('int')
    medal_df['Silver'] = medal_df['Silver'].astype('int')
    medal_df['Bronze'] = medal_df['Bronze'].astype('int')
    medal_df['Total'] = medal_df['Total'].astype('int')




    return medal_df

def country_year_list(df):
    years = df['Year'].unique().tolist()
    years.sort()
    years.insert(0, 'Overall')

    new_df=df
    new_df['region']=df['region'].astype('str')
    country = np.unique(new_df['region'].values).tolist()
    country.sort()
    country.insert(0, 'Overall')

    return years,country

def data_over_time(df,col):

    nations_over_time = df.drop_duplicates(['Year', col])['Year'].value_counts().reset_index().sort_values('index')
    nations_over_time.rename(columns={'index': 'Edition', 'Year': col}, inplace=True)
    return nations_over_time

def most_successful(df, sport):
    temp_df = df.dropna(subset=['Medal'])


    if sport != 'Overall':
        temp_df = temp_df[temp_df['Sport'] == sport]

   # x = temp_df['Name'].value_counts().reset_index().head(15).merge(df, left_on='index', right_on='Name', how='left')[
    #    ['index', 'Name_x', 'Sport', 'region']].drop_duplicates('index')
   # x.rename(columns={'index': 'Name', 'Name_x': 'Medals'}, inplace=True)
    x = temp_df.groupby('Name').sum()[['Gold', 'Silver', 'Bronze']].sort_values(['Gold', 'Silver', 'Bronze'],ascending=False).head(15)
    samp_df = df[['Name', 'Sport', 'region','Medal']].dropna(subset=['Medal'])
    x = pd.merge(x, samp_df, on='Name', how='left')


    x['Total']=x['Gold']+ x['Silver'] +x['Bronze']
    x=x[['Name','region','Sport','Gold','Silver','Bronze','Total']].drop_duplicates(['Name']).sort_values(['Gold', 'Silver', 'Bronze'],ascending=False).head(15)

    return x

def yearwise_medal_tally(df,country):
    temp_df = df.dropna(subset=['Medal'])
    temp_df.drop_duplicates(subset=['Team', 'NOC', 'Games', 'Year', 'City', 'Sport', 'Event', 'Medal'], inplace=True)

    new_df = temp_df[temp_df['region'] == country]
    final_df = new_df.groupby('Year').count()['Medal'].reset_index()

    return final_df

def country_event_heatmap(df,country):
    temp_df = df.dropna(subset=['Medal'])
    temp_df.drop_duplicates(subset=['Team', 'NOC', 'Games', 'Year', 'City', 'Sport', 'Event', 'Medal'], inplace=True)

    new_df = temp_df[temp_df['region'] == country]

    pt = new_df.pivot_table(index='Sport', columns='Year', values='Medal', aggfunc='count').fillna(0)
    return pt


def most_successful_countrywise(df, country_select) :
    temp_df = df.dropna(subset=['Medal'])

    temp_df=temp_df[temp_df['region']==country_select]

    # x = temp_df['Name'].value_counts().reset_index().head(15).merge(df, left_on='index', right_on='Name', how='left')[
    #    ['index', 'Name_x', 'Sport', 'region']].drop_duplicates('index')
    # x.rename(columns={'index': 'Name', 'Name_x': 'Medals'}, inplace=True)
    x = temp_df.groupby('Name').sum()[['Gold', 'Silver', 'Bronze']].sort_values(['Gold', 'Silver', 'Bronze'],
                                                                                ascending=False).head(15)
    samp_df = df[['Name', 'Sport', 'region', 'Medal']].dropna(subset=['Medal'])
    x = pd.merge(x, samp_df, on='Name', how='left')

    x['Total'] = x['Gold'] + x['Silver'] + x['Bronze']
    x = x[['Name', 'Sport', 'Gold', 'Silver', 'Bronze', 'Total']].drop_duplicates(['Name']).sort_values(
        ['Gold', 'Silver', 'Bronze'], ascending=False).head(10)

    return x


def weight_v_height(df,sport):
    athlete_df = df.drop_duplicates(subset=['Name', 'region'])
    athlete_df['Medal'].fillna('No Medal', inplace=True)
    if sport != 'Overall':
        temp_df = athlete_df[athlete_df['Sport'] == sport]
        return temp_df
    else:
        return athlete_df

def men_vs_women(df):
    athlete_df = df.drop_duplicates(subset=['Name', 'region'])

    men = athlete_df[athlete_df['Sex'] == 'M'].groupby('Year').count()['Name'].reset_index()
    women = athlete_df[athlete_df['Sex'] == 'F'].groupby('Year').count()['Name'].reset_index()

    final = men.merge(women, on='Year', how='left')
    final.rename(columns={'Name_x': 'Male', 'Name_y': 'Female'}, inplace=True)

    final.fillna(0, inplace=True)

    return final




