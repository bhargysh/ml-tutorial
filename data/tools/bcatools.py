import numpy as np
import pandas as pd

from math import cos, sin, asin, sqrt, atan2, pi

from IPython.display import display, HTML

####################
def show(df, n=5):
    display(HTML(df[:n].to_html()))
    
def ls(d):
    return [x for x in os.listdir(d) if x != '.DS_Store']

def count_many_cols(df, id_col):
    """
    perform a count over many columns which share a close value space. E.g., bed, bath, car.
    """
    dfm = pd.melt(
    df, 
    id_vars=id_col,
    var_name="attribute",
    value_name='count')

    dfm = dfm.groupby(['attribute', 'count']).count().unstack('attribute').fillna(0)
    return dfm


######################

def haversine_distance(loc1, loc2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    loc is a tuple or list of (latitude, longitude).
    
    """
    # convert decimal degrees to radians 
    lat1, lon1, lat2, lon2 = loc1[0], loc1[1], loc2[0], loc2[1]
    lat1, lon1, lat2, lon2 = map(lambda d: d * pi/180, [lat1, lon1, lat2, lon2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r_earth = 6371   # Radius of earth in kilometers. Use 3956 for miles
    return c * r_earth

def bearing(loc1, loc2):
    """
    Calculate the bearing from one location to another.
    loc is a tuple or list of (latitude, longitude).
    """
    lat1, lon1, lat2, lon2 = loc1[0], loc1[1], loc2[0], loc2[1]
    lat1, lon1, lat2, lon2 = map(lambda x: x*pi/180, [lat1, lon1, lat2, lon2])
    
    x = sin(lon2 - lon1)*cos(lat2)
    y = cos(lat1)*sin(lat2) - sin(lat1)*cos(lat2)*cos(lon2 - lon1)
    
    bearing = (atan2(x,y) % (2*pi)) * (180/pi)
    return bearing


def sigfig(x, n):
    return np.round(x, decimals=-(int(np.log10(x))-n+1))

#######################################################

def num_formatter(prec):
    def fr(x):
        naive = '{:.{prec}f}'.format(x, prec=prec)
        whole, dec = naive.split('.')
        cut = dec.find('0')
        if cut == 0:
            return whole
        else:
            return "{}.{}".format(whole,dec[:cut])
    return fr
#########################################################