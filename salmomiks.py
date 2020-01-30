
import csv
import pandas as pd
import folium
import glob
import numpy as np
import matplotlib.pyplot as plt
import folium

def load_data(recievers_list_file
              , observations_filelist
              , filtered_data_file
):

    #TODO make sure correct results on different systems
    print('Processing receiver list. This is for NESS system only for now')
    receiver_list = pd.read_csv(recievers_list_file)
    print('Removing records with invalid receiver numbers')
    print(receiver_list[receiver_list['id'].isnull()])
    receiver_list = receiver_list.dropna()

    if len(receiver_list['id']) != len(receiver_list['id'].unique()):
        print('WTF! Multiple receivers with the same ID! Will use the top value of GPS')
        #print(receiver_list[receiver_list['id'].duplicated()])
        print(pd.concat(g for _, g in receiver_list[receiver_list['system']=='Ness'].groupby("id") if len(g) > 1)) #TODO NESS

    print('Processing observations list')

    #Adding receiver info and GPS to each receiver dataframe
    #Possibly a typo in reciever list 483460 is 483466 TODO, check important
    observations = []
    for f in observations_filelist:
        rec_id = int(f.split('_')[1])
        df = pd.read_csv(f)
        df['rec_id']=rec_id
        print(rec_id)
        select_filter = receiver_list['id']==rec_id
        if (select_filter).any():
            df['lat']=receiver_list[select_filter]['lat'].iloc[0]
            df['lon']=receiver_list[select_filter]['lon'].iloc[0]
            df['system']=receiver_list[select_filter]['system'].iloc[0]
        else:
            print('WTF! Receiver ' + str(rec_id) + ' not on receiver list!')
        observations.append(df)


    print('Concatenating the dataframe')
    df_observations = pd.concat(observations)
    df_observations = df_observations.drop(["Latitude","Longitude","Receiver","Sensor Unit","Sensor Value","Station Name","Transmitter Name","Transmitter Serial"], axis=1)
    print(df_observations.head(10))

    df_observations.columns = ['datetime', 'transmitter','lat','lon','rec_id','system']
    #check if I have all receivers in receiver list
    #not the moste efficent way ;)
    rec_in_obs = df_observations['rec_id'].unique()
    rec_on_file = receiver_list[receiver_list['system']=='Ness']['id'].unique() #TODO only NESS
    missing_observations = list(set(rec_on_file) - set(rec_in_obs))
    print('Missing Observations')
    print(missing_observations)


    # Check if receiver ID matches optional #TODO
    #,Date and Time (UTC),Receiver,Transmitter,Transmitter Name,Transmitter Serial,Sensor Value,Sensor Unit,Station Name,Latitude,Longitude,rec_id
    #0,2019-04-10 14:03:59,VR2Tx-483479,A69-1601-63479,,,,,,,,483479

    #check for the same fish in multiple systems, optional #TODO




    print('Saving observations to filtered.csv')
    df_observations.to_csv(filtered_data_file)

    return None

def make_rec_map(filtered_file, map_file):


    filtered_list = pd.read_csv(filtered_file)
    fish_counts = filtered_list.groupby('rec_id')['transmitter'].count()
    rec_list = filtered_list.drop_duplicates(['rec_id']).set_index('rec_id')
    rec_list = rec_list.dropna()

    m = folium.Map(location=[57.0890, -4.7483], zoom_start=15)
    for index, rec in rec_list.iterrows():
        colour = 'red'
        fishes = fish_counts[index]

        folium.CircleMarker(location = [rec['lat'], rec['lon']]
                            , popup=str(index)
                            , color = colour
                            , radius=np.sqrt(fishes)/10
                            , fill=True
                            , fill_color=colour
                            , stroke = True
                            , weight=1
                            , fill_opacity=.55
        ).add_to(m)
        m.save(map_file)

def make_fish_df(filtered_file, fish_file):
    filtered_list = pd.read_csv(filtered_file)
