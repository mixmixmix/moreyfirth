import pandas as pd
import glob
import csv
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from itertools import cycle
import seaborn as sns
import pickle
import dateutil
import folium
import pyproj



newf = pd.read_csv('indata/mf/Salmon.csv')

ness = newf[newf['array']=='Ness']

recids = ness['receiver'].unique()
smoltids = ness['tagid'].unique()

df_observations = ness


#Adding distance from first reciever
first_rec_id=480432
opposite_dir_id=483479 # this is a reciever at the wrong end of a lake
rec_list = df_observations.drop_duplicates(['receiver']).set_index('receiver')
rec_list = rec_list.dropna()
print(rec_list)
geod = pyproj.Geod(ellps='WGS84')
dists=dict()
for index, rec in rec_list.iterrows():
    if index == first_rec_id:
        dist = 0
    else:
        _, _,dist = geod.inv(  rec_list.loc[first_rec_id]['longitude']
                            , rec_list.loc[first_rec_id]['latitude']
                            , rec['longitude']
                            , rec['latitude'])
    if index == opposite_dir_id:
        dist = - dist

    dists[index]=int(dist)

ness['dists']=ness['receiver'].replace(dists)

myids = smoltids[0:-1]
for myid in myids:
    plt.plot(ness[ness['tagid']==myid]['datetime'],ness[ness['tagid']==myid]['dists'])
plt.show()


#at each receiver there is a 
rid = 483488
ness['datetime']=pd.to_datetime(ness['datetime'])
restab = dict()
for rid in recids:
    print(f'Rec {rid}')
    if rid == 480432:
        continue
    restab[rid] = -1*np.ones((len(smoltids),len(smoltids)),dtype=int)
    for i, smoltid in enumerate(smoltids):
        csmolt = ness[(ness['receiver']==rid) &(ness['tagid']==smoltid)]['datetime']
        print(f'smoltt {i}')
        for j,smoltid2 in enumerate(smoltids):
            if j > i:
                dsmolt = ness[(ness['receiver']==rid) & (ness['tagid']==smoltid2)]['datetime']
                mds = recmindist(csmolt,dsmolt)
                # print(f'and {i}')
                restab[rid][i,j]=mds.seconds

def recmindist(a,b):
    tdiff = datetime.timedelta(weeks=100)
    for ela in a:
        for elb in b:
            tsign = ela-elb
            if tsign.seconds < 0: #all the following numbers wont count
                break
            tdiffn = abs(tsign)
            if tdiffn < tdiff:
                tdiff = tdiffn
    return tdiff

def save_obj(obj, name ):
    with open('data/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('data/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)





a = restab[480432]
sns.heatmap(a, vmin = 0, vmax=600); plt.show()

# close file
f.close()

    for each observation 

print()
# find a minimal time

reclocs = list()
for ri in recids:
    lat = newf[newf['receiver']==ri].loc[0]['latitude']
    lon = newf[newf['receiver']==ri].loc[0]['longitude']
    reclocs.append((lat,lon))


