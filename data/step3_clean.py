#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys, os
# Handy functions
sys.path.append("/app/tools")
import bcatools as bt
import bcaplots as bp

import numpy as np
import pandas as pd
pd.set_option('display.width', 5000) 
pd.set_option('display.max_columns', 60)
pd.set_option('display.float_format', bt.num_formatter(3))

from matplotlib import pyplot as plt
from matplotlib.ticker import LogFormatterExponent
import seaborn as sns


# In[ ]:


export_loc = '/app/data/processed/sold_listings_with_station.csv'
df_listings = pd.read_csv(export_loc)


# ## Get some basic stats for the columns in the dataset
# 
# mean, median, etc.
# pandas has a neat function `.describe()` that does this for you.

# In[ ]:


df_listings.describe()


# ## Clean some variables up!
# 
# - `sold_price`
# - `landsize`
# - `buildingsize`
# - `garages`
# - `carports`
# - `bedrooms`
# - `bathrooms`
# - `toilets`
# 

# ### start with `sold_price`

# Sold price is heavily skewed. we can adjust for this skewness by taking the logarithm of this data.

# In[ ]:


df_listings['log_price'] = np.log(df_listings.sold_price)
# make a function to convert the axes values back to the real numbers
plot_ticker = lambda x: bp.millify(np.exp(x))

bp.hist(df_listings.log_price, n_bins=200, tick_func=plot_ticker)
fig = plt.gcf()
fig.suptitle('Histogram of property prices')
plt.xlabel('Property price')
plt.ylabel('Frequency of listings')
plt.show()


# In[ ]:


df_listings.loc[df_listings.sold_price == 30000]


# In[ ]:


cleaned = df_listings.copy()


# In[ ]:


# delete rows where sold price is less that 100K
#get those rows
rows_to_delete = cleaned[cleaned.sold_price < 100000]
cleaned = cleaned.drop(rows_to_delete.index)


# In[ ]:


# replot
bp.hist(cleaned.log_price, n_bins=200, tick_func=plot_ticker)
plt.show()


# ### landsize

# In[ ]:


bp.hist(df_listings.landsize, n_bins=100)
plt.show()


# In[ ]:


# DROP FIRST AND LAST PERCENTILE
s = cleaned.landsize.sort_values()
N = int(s.shape[0]*0.01)

cleaned = cleaned.drop(s[:N].index)
cleaned = cleaned.drop(s[-N:].index)


# In[ ]:


bp.hist(df_listings.garages, n_bins=20)
fig = plt.gcf()
fig.suptitle('Histogram of car spots at each house')
plt.xlabel('Number of car spots')
plt.ylabel('Frequency of listings')


# In[ ]:


df_listings[df_listings.garages > 20]


# In[ ]:


# REPLOT
bp.hist(cleaned.landsize, n_bins=100)
plt.show()


# Explore the relationship between price and distance from cbd. you can use our prettified scatter plot __`bp.facet_scatter()`__

# In[ ]:


bp.facet_scatter(cleaned, 
              xcol='dist_from_cbd', 
              ycol='log_price', 
              facet='property_type', 
              trendline=True,
              trend_order=2,
              tick_func=plot_ticker
            )
plt.show()


# In[ ]:


z = np.polyfit(cleaned.dist_from_cbd, cleaned.log_price, 2)
p = np.poly1d(z)

line_eq = "y = {:.6f}x^2 + {:.6f}x + {:.6f}".format(*z)
# plt.plot(x,p(x), '-', color=colours[N+n], label="trend {: <12} {}".format(ptype, line_eq))
# " + ".join()


# In[ ]:


line_eq


# ## `buildingsize`

# In[ ]:


bp.hist(cleaned.buildingsize, n_bins=20)


# In[ ]:


# DROP FIRST AND LAST PERCENTILE
s = cleaned.buildingsize.sort_values()
N = int(s.shape[0]*0.01)

cleaned = cleaned.drop(s[:N].index)
cleaned = cleaned.drop(s[-N:].index)


# In[ ]:


bp.hist(cleaned.buildingsize, n_bins=20)


# Main Attributes
# Let's look at a count of the values and look at reasonable cut-offs

# In[ ]:


bt.count_many_cols(
    cleaned[["bedrooms", 
             "bathrooms", 
             "toilets", 
             "garages", 
             "carports", 
             "listing_id"]], 
    id_col='listing_id')


# In[ ]:


cut_offs = {
    'bedrooms': 5,
    'bathrooms': 4,
    'toilets': 4,
    'carports': 4,
    'garages': 4,
}


# In[ ]:


for att, val in cut_offs.items():
    x = cleaned[att] > val
    cleaned.loc[x, att] = val


# In[ ]:


bt.count_many_cols(
    cleaned[["bedrooms", 
             "bathrooms", 
             "toilets", 
             "garages", 
             "carports", 
             "listing_id"]], 
    id_col='listing_id')


# # Price vs bedrooms
# what about against a categorical variable like bedrooms. how does price correlate to that?

# In[ ]:


cleaned.boxplot(column="log_price", by="bedrooms", figsize=(10,6))
plt.show()


# In[ ]:


r = cleaned.dist_from_cbd
theta = cleaned.dir_from_cbd

# convert theta to radians
# North in geocoord is 0, and the angle goes in a clockwise direction, but, for radians,
# 0 points East, and the angle goes in a counter-clockwise direction. 
# you'll need to convert the geoocords into radians.

theta_rad = list(map(lambda x: np.pi/2-x*(np.pi/180), theta))

x = r*np.cos(theta_rad)
y = r*np.sin(theta_rad)

bp.hex_plot(x, y)


# # Hexplot of price

# In[ ]:


r = cleaned.dist_from_cbd
theta = cleaned.dir_from_cbd

# convert theta to radians
# North in geocoord is 0, and the angle goes in a clockwise direction, but, for radians,
# 0 points East, and the angle goes in a counter-clockwise direction. 
# you'll need to convert the geoocords into radians.

theta_rad = list(map(lambda x: np.pi/2-x*(np.pi/180), theta))

x = r*np.cos(theta_rad)
y = r*np.sin(theta_rad)

bp.hex_plot(x, y, z=df_listings.log_price, tick_func=plot_ticker)


# In[ ]:


cleaned.sold_price.min()


# In[ ]:


cleaned = cleaned[cleaned.sold_price >= 200000]


# In[ ]:


cleaned.to_csv('/app/data/processed/sold_listings_cleaned.csv', index=False)


# ## price vs year built

# In[ ]:


df = cleaned.sort_values("year_built")[100:]


# In[ ]:


bp.facet_scatter(df, xcol="year_built", ycol="log_price", facet='property_type', tick_func=plot_ticker)
plt.show()

