import pandas as pd

def type_select(df,region_df,selected):

    if selected == 'Summer':

        df = df[df['Season'] == 'Summer']

    if selected == 'Winter' :
        df = df[df['Season'] == 'Winter']

    if selected == 'Combined' :
        df = df



    df= df.merge(region_df, on='NOC',how='left')

    df.drop_duplicates(inplace=True)

    df=pd.concat([df,pd.get_dummies(df['Medal'])],axis=1)

    return df

