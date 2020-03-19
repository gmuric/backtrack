# backtracking infected people using geo-location to identify risky encounters
# Author: Goran Muric


import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import math
import warnings
warnings.filterwarnings('ignore')

def distance(origin, destination):
    # Haversine formula example in Python
    # Author: Wayne Dyck
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371 # km

    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c

    return (d)

def get_risky(data, target_id, max_radius=100, time_window=20):
    timed = datetime.timedelta(days=0, seconds=0, microseconds=0, milliseconds=0, minutes=time_window, hours=0, weeks=0)
    cand_dict = {}

    primary_path = data[data['id']==target_id]
    candidates=pd.DataFrame()
    for row_id,row in primary_path.iterrows():
        candidate = data[(data['time']>row.loc['time']-timed) & (data['time']<row.loc['time']+timed) & (data['id']!=target_id)]
        distances = []
        for r_id,r in candidate.iterrows():
            distances.append(distance((row['lat'],row['lng']), (r['lat'],r['lng']))*1000)
        candidate.loc[:,'dist'] = distances
        candidate = candidate[candidate['dist']<=max_radius]
        candidates = pd.concat([candidates,candidate])

    if len(candidates)==0:
        return (None)

    for row_id, row in candidates.drop_duplicates(subset='id').iterrows():
        cand_dict[row.loc['id']] = {}

    for row_id, row in candidates.groupby('id')['dist'].min().to_frame().iterrows():
        cand_dict[row_id]['min_dist'] = row.loc['dist']
        cand_dict[row_id]['lat_point'] = candidates[candidates['dist']==row.loc['dist']].iloc[0].loc['lat']
        cand_dict[row_id]['lng_point'] = candidates[candidates['dist']==row.loc['dist']].iloc[0].loc['lng']

    for row_id, row in candidates.groupby('id')['dist'].count().to_frame().iterrows():
        cand_dict[row_id]['num_encounters'] = row.loc['dist']

    times = {}
    g = candidates.groupby('id')
    for name, group in g:
        time = datetime.timedelta(0)
        for row_id, row in group.iterrows():
            t = row.loc['time']
            d1 = data[(data['id']==target_id) & (data['time']<t+timed) & (data['time']>t-timed)]
            d2 = data[(data['id']==row.loc['id']) & (data['time']<t+timed) & (data['time']>t-timed)]
            d = pd.concat([d1,d2])
            time = time + (d1['time'].max() - d1['time'].min())
        times[name] = time.seconds

    for key, value in cand_dict.items():
        cand_dict[key]['duration'] = times[key]

    res = pd.DataFrame.from_dict(cand_dict).T
    res['min_dist_inverse'] = [1/x if x > 0 else 1/(x+0.01) for x in list(res['min_dist'])]

    normalize_cols = ['duration', 'num_encounters']
    res_norm = pd.DataFrame()
    for col in normalize_cols:
        lower = res[col].min()/res[col].max()
        upper = 1
        res_norm[col] = [(upper - lower) * (x - res[col].min())/(res[col].max()-res[col].min()) + lower for x in list(res[col])]
 
        
    res = res.reset_index().rename(columns={'index':'id'})
    
    for col in [c for c in res.columns if c not in normalize_cols]:
        res_norm[col] = res[col].values
    
    res_norm['score'] = res_norm['duration']*res_norm['min_dist_inverse']*res_norm['num_encounters']

    res = res.join(res_norm['score']).drop(columns=['min_dist_inverse'])
    return(res.to_dict('records'))