
import glob
import csv
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

import dateutil
import folium
import pyproj

def load_data(recievers_list_file
              , tag_list_file
              , observations_filelist
              , filtered_data_file
):

    #TODO make sure correct results on different systems
    print('Processing receiver list. This is for NESS system only for now')
    receiver_list = pd.read_csv(recievers_list_file)
    print('Processing tag list. This is for NESS system only for now')
    tag_list = pd.read_csv(tag_list_file)
    tag_list = pd.read_csv(tag_list_file)
    tag_list = tag_list[tag_list['River']=='Ness']['Tag_ID'].reset_index(drop=True) #TODO add predator tag info
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

        #60/66 fix?HACK???
        #if rec_id == 483466:
        #    select_filter = receiver_list['id']==483460

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

    #Remove alien tags
    print('Removing records with invalid tag numbers')
    df_observations['tags'] = df_observations.apply(lambda x: int(x['transmitter'].split('-')[-1]),axis=1)

    tags_in_system = tag_list.values
    transmitters_in_system = df_observations['tags'].unique()
    #print(tags_in_system)
    #print(transmitters_in_system)
    print('Removing {} alien signals'.format(len(list(set(transmitters_in_system)-set(tags_in_system)))))
    df_observations = df_observations[df_observations['tags'].isin(tag_list.values)]

    print('Calculate distances')
    print(df_observations.sample(30))
    #Adding distance from first reciever
    first_rec_id=480432
    rec_list = df_observations.drop_duplicates(['rec_id']).set_index('rec_id')
    rec_list = rec_list.dropna()
    print(rec_list)
    geod = pyproj.Geod(ellps='WGS84')
    dists=dict()
    for index, rec in rec_list.iterrows():
        if index == first_rec_id:
            dist = 0
        else:
            _, _,dist = geod.inv(  rec_list.loc[first_rec_id]['lon']
                                , rec_list.loc[first_rec_id]['lat']
                                , rec['lon']
                                , rec['lat'])
        dists[index]=int(dist)

    rec_list['dist_first']=pd.Series(dists)
    df_observations['dist_first'] = df_observations['rec_id'].replace(dists)
    df_observations['epoch'] = df_observations.apply (lambda row: dateutil.parser.isoparse(row['datetime']).timestamp(), axis=1)

    #Adding time in 0.1 of hours
    epoch_beginning = 1552413619.0
    df_observations['epoch'] = df_observations.apply (lambda row: ((row['epoch']-epoch_beginning)/360), axis=1) #epoch is the time in 0.1 of hours from the first ping ever on NESS

    df_observations = df_observations.sort_values(by=['epoch']).reset_index(drop=True)

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

#This function is NESS specific for now (TODO)
def make_fish_dist(filtered_file):
    #first receiver id
    filtered_list = pd.read_csv(filtered_file).dropna()
    filtered_list = filtered_list.sort_values(by=['epoch']).reset_index(drop=True)

    tags = filtered_list['tags'].unique()
    print('Printing graphs...')

    for tx in tags:
        ff = filtered_list[filtered_list['tags']==tx].reset_index(drop=True)
        if(len(ff)<10):
            continue
        ax = sns.countplot(x="dist_first", data=ff)
        ax.set_xticklabels(ax.get_xticklabels(),rotation=30)
        ax.set_title(str(tx))
        plt.savefig('data/output/distcount'+str(tx)+'png')
        plt.clf()
        #ay = sns.distplot(ff['epoch'].values)
        ay = sns.scatterplot(x="epoch", y='dist_first', data=ff)
        #ay.set_xticklabels(ay.get_xticklabels(),rotation=90)
        ay.set_title(str(tx))
        plt.savefig('data/output/timedist'+str(tx)+'png')
        plt.clf()

    print('done')

def make_fish_corr(filtered_file):
    #first receiver id
    filtered_list = pd.read_csv(filtered_file).dropna()
    filtered_list = filtered_list.sort_values(by=['epoch']).reset_index(drop=True)

    tags = filtered_list['tags'].unique()
    #calculate correlation between different fish
    #order fish by 'distance traveled'
    #order fish by ho fast to last rec
    correg=np.zeros((len(tags),len(tags)))
    for idx, tx in enumerate(tags):
        fx = filtered_list[filtered_list['tags']==tx].set_index('epoch')
        for idy, ty in enumerate(tags):
            fy = filtered_list[filtered_list['tags']==ty].set_index('epoch')
            correg[idx,idy] =  fx['dist_first'].corr(fy['dist_first'])


    ###############PRINT MAIN MAP
    colormap = cm.get_cmap('inferno')
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(correg,vmin=-1, vmax=1, cmap=colormap)
    ax.set_axis_off()
    fig.tight_layout()

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = mpl.colorbar.ColorbarBase(cbar_ax, cmap=colormap,norm=mpl.colors.Normalize(vmin=-1, vmax=1))

    fig.suptitle("correlation)",fontsize=14)
    fig.savefig('data/output/' + 'correlation.png')
    #plt.show()


    print("done!")

def make_fish_speed(filtered_file, fish_file):
    dfish = pd.read_csv(filtered_file).dropna()
    dfish = dfish.sort_values(by=['epoch']).reset_index(drop=True)
    tags = dfish['tags'].unique()

    allfish = list()

    for tx in tags:
        ff = dfish[dfish['tags']==tx].reset_index(drop=True)
        ff['t_diff']=ff['epoch'] - ff['epoch'].shift(1)
        ff['t_dist']=ff['dist_first'] - ff['dist_first'].shift(1)
        ff['speed']=ff.apply(lambda row: row['t_dist']/(360*row['t_diff']), axis=1)
        ff=ff[ff['speed']!=0] #remove uninteresing stuff
        if len(ff) == 0:
            continue
        ay = sns.scatterplot(x="datetime", y='speed', data=ff)
        ay.set_title(str(tx))
        ay.set_xticklabels(ay.get_xticklabels(),rotation=45)
        plt.savefig('data/output/speed'+str(tx)+'png')
        plt.clf()
        allfish.append(ff)

    fishmove = pd.concat(allfish)
    fishmove.to_csv(fish_file)


#USE THIS TO CALCULATE SPEED
#import dateutil
#timethreshold = 30 #how many seconds max difference between pings from a fish
#dffilts = dict()
#for key, df in dfs.items():
#    df['tesc'] = df.apply (lambda row: dateutil.parser.isoparse(row['Date and Time (UTC)']).timestamp(), axis=1)
#    df['datetime'] = df.apply (lambda row: dateutil.parser.isoparse(row['Date and Time (UTC)']), axis=1)
#    df = df [['tesc','datetime','Transmitter']]
#    #df.set_index('tesc')
#    df['fishdiff']= ~(df['Transmitter'] == df['Transmitter'].shift())
#    df['mfishdiff']= ~(df['Transmitter'].shift(-1) == df['Transmitter'])
#    df['tdiff']=df['tesc'] - df['tesc'].shift(1, fill_value=1554901439)
#    df['mtdiff']=df['tesc'].shift(-1, fill_value=1554901439) - df['tesc']
#    dffilt = df[((df['tdiff']<timethreshold) & (df['fishdiff'])) | ((df['mtdiff']<timethreshold)  & (df['mfishdiff']))]
#    plt.plot(dffilt['datetime'],dffilt['tdiff'],'o')
#    plt.show()
#    display(dffilt.head(10))
#    dffilts[key]=dffilt
#
#In [ ]:
##give clusters a name
#for dffilt in dffilts.values():
#    groupid = 0
#    shoals = []
#    for ix, valx in dffilt.iterrows():
#        if valx['tdiff']>=timethreshold:
#            groupid=groupid+1
#        shoals.append(groupid)
#        
#    dffilt['shoals']=shoals        
#    plt.plot(dffilt['datetime'],dffilt['shoals'],'o')
#    plt.show()
#
##plot cluster sizes x cluster number per time
##show that on a map
#
#In [ ]:
#for key, dffilt in dffilts.items():
#    print(key)
#    plt.plot(dffilt.groupby('shoals').size())
#    plt.show()
#
#In [ ]:
#
#
#
#
