#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys, os
# Handy functions
sys.path.append("/app/tools")
import bcatools as bt
import bcaplots as bp

import numpy as np
import pandas as pd
pd.set_option('display.width', 5000) 
pd.set_option('display.max_columns', 60)
# pd.set_option('display.float_format', bt.num_formatter(3))
pd.set_option('display.float_format', lambda x: "{:.3f}".format(x))

from matplotlib import pyplot as plt
from matplotlib.ticker import LogFormatterExponent
import seaborn as sns
from xgboost import XGBRegressor, plot_importance, DMatrix


# In[2]:


def rmse(actual, predicted, log_price=True):
    if log_price:
        actual = np.exp(actual)
        predicted = np.exp(predicted)
    error = actual - predicted
    return np.sqrt(np.mean(error ** 2))

error_dict = {}


# In[3]:


export_loc = '/app/data/processed/sold_listings_cleaned.csv'
df_listings = pd.read_csv(export_loc)


# In[4]:


pd.get_dummies(df_listings.property_type, drop_first=True)


# ## Linear Regression

# In[5]:


from sklearn.linear_model import LinearRegression


# # select variables

# In[6]:


df_listings[:5]


# In[7]:


features = [
    'property_type', 
    'landsize', 
    'buildingsize', 
    'garages', 
    'carports', 
    'bedrooms', 
    'bathrooms', 
    'toilets', 
    'dist_from_cbd', 
    'dir_from_cbd', 
    "log_price",
    'dist_to_station'
]


# In[8]:


df = df_listings[features].copy()


# In[9]:


df = pd.get_dummies(df, columns=['property_type'], drop_first=True)


# In[10]:


df['land_building_ratio'] = df.landsize / df.buildingsize
df = df[df.land_building_ratio <= 10]
# df = df[(df.yardsize <= 1000) & (df.yardsize >= 0)]


# In[11]:


df.land_building_ratio.plot.hist(figsize=(12,8), bins=50)


# Make car spaces variable

# In[12]:


df["carspaces"] = df["garages"] + df["carports"]


# ## One hot encode variables

# In[13]:


df[:5]


# In[14]:


cols_to_encode = [
    "property_type", 
    "bedrooms", 
    "bathrooms", 
    "toilets", 
    "garages", 
    "carports", 
    "carspaces"
                 ]


# Split training and test data

# In[15]:


from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

train_x, test_x, train_y, test_y = train_test_split(df.drop('log_price', axis=1),
                                                        df.log_price,
                                                        train_size=0.8, random_state=1234)


# In[16]:


print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)


# In[17]:


lreg = LinearRegression(n_jobs=4, normalize=False)


# In[18]:


lreg.fit(train_x, train_y)


# In[19]:


performance = {}

performance["train"] = lreg.score(train_x, train_y)

performance['test'] = lreg.score(test_x, test_y)

for k,v in performance.items():
    print("{} performance R^2 = {}".format(k,v))


# In[20]:


error_dict['Basic linear regression'] = rmse(test_y, lreg.predict(test_x), log_price=True)
error_dict


# In[21]:


pred = np.exp(lreg.predict(test_x))
actual = np.exp(test_y)

compare = pd.DataFrame(list(zip(pred, actual)), columns=["predicted", "actual"])
compare["abs_diff"] = np.abs(compare["predicted"] - compare["actual"])
compare["err"] = compare["abs_diff"]/compare["actual"]

compare.set_index(test_x.index).join(df_listings[features + ['suburb']])


# In[22]:


from itertools import combinations_with_replacement

def non_linearise(M, order=2):
    
    def level_up(res, M, levels_to_go):
        if levels_to_go <=1:
            return res
        else:
            combs = list(combinations_with_replacement(range(res.shape[1]),2))
            prod = np.array([
                [res[k, i]*res[k,j] for i,j in combs]
                for k in range(res.shape[0])
            ])
            out = np.concatenate([M,prod], axis=1)
            return level_up(prod, M, levels_to_go - 1)
        
    return level_up(M, M, order)


# In[23]:


def nl(data):
    return non_linearise(data.values, 2)


# In[24]:


X = train_x.values
Y = train_y.values
X2 = non_linearise(X,2)


# In[25]:


lreg2 = LinearRegression(n_jobs=4, normalize=False)


# In[26]:


lreg2.fit(X2, Y)


# In[27]:


performance = {}

performance["train"] = lreg2.score(X2, Y)

performance['test'] = lreg2.score(non_linearise(test_x.values, 2), test_y)

for k,v in performance.items():
    print("{} performance R^2 = {}".format(k,v))


# In[28]:


error_dict['Linear regression with interaction effects'] = rmse(test_y, lreg2.predict(non_linearise(test_x.values)))
error_dict


# In[29]:


pred = np.exp(lreg2.predict(non_linearise(test_x.values)))
actual = np.exp(test_y)

compare = pd.DataFrame(list(zip(pred, actual)), columns=["predicted", "actual"])
compare["abs_diff"] = np.abs(compare["predicted"] - compare["actual"])
compare["err"] = compare["abs_diff"]/compare["actual"]

compare.set_index(test_x.index).join(df_listings[features])


# In[30]:


from sklearn.ensemble import BaggingRegressor


# In[31]:


bgr1 = BaggingRegressor(base_estimator=LinearRegression(n_jobs=1),
                      n_estimators=50,
                      max_samples=0.6,
                      max_features=1.0,
                      oob_score=True,
                       n_jobs=4
              )


# In[32]:


bgr1.fit(train_x, train_y)


# In[33]:


performance = {}

performance["train"] = bgr1.score(train_x, train_y)

performance['test'] = bgr1.score(test_x, test_y)

for k,v in performance.items():
    print("{} performance R^2 = {}".format(k,v))


# In[34]:


error_dict['Simple Bagging regressor'] = rmse(test_y, bgr1.predict(test_x), log_price=True)
error_dict


# In[35]:


bgr2 = BaggingRegressor(base_estimator=LinearRegression(n_jobs=1),
                      n_estimators=50,
                      max_samples=0.6,
                      max_features=1.0,
                      oob_score=True,
                       n_jobs=4,
                        verbose=1
              )


# In[36]:


bgr2.fit(X2,Y)


# In[37]:


performance = {}

performance["train"] = bgr2.score(X2, Y)

performance['test'] = bgr2.score(non_linearise(test_x.values, 2), test_y)

for k,v in performance.items():
    print("{} performance R^2 = {}".format(k,v))


# In[38]:


error_dict['Bagging regressor with interaction effects'] = rmse(test_y, bgr2.predict(non_linearise(test_x.values, 2)))
error_dict


# In[39]:


xtrain_x, xtest_x, xtrain_y, xtest_y = train_test_split(train_x, train_y, random_state=1234, train_size=0.8)


# In[40]:


xgb1 = XGBRegressor(n_estimators=400, learning_rate=0.05, seed=1234)


# In[41]:


xgb1.fit(xtrain_x, xtrain_y, early_stopping_rounds=25, eval_metric='rmse', eval_set=[(xtest_x, xtest_y)])


# In[42]:


xgb1.score(test_x, test_y)


# ### XGBoost was really interesting because it would have made its own binary splits on the direction from the CBD rather than treating it as a number to multiply (eg if direction < 180 and direction > 90, do this)

# In[43]:


plot_importance(xgb1)
fig = plt.gcf()
fig.set_size_inches(10,15)


# In[44]:


error_dict['Simple XGBoost model'] = rmse(test_y, xgb1.predict(test_x), log_price=True)
error_dict


# In[45]:


xgb2 = XGBRegressor(n_estimators=400, learning_rate=0.05, seed=1234)


# In[46]:


xgb2.fit(nl(xtrain_x), xtrain_y, early_stopping_rounds=25, eval_metric='rmse', eval_set=[(nl(xtest_x), xtest_y)])


# In[47]:


error_dict['XGBoost model with interactions'] = rmse(test_y, xgb2.predict(nl(test_x)), log_price=True)
error_dict


# In[48]:


train_x, test_x, train_y, test_y = train_test_split(df.drop('log_price', axis=1),
                                                    df.log_price,
                                                    train_size=0.8,
                                                    random_state=1234)


xtrain_x, xtest_x, xtrain_y, xtest_y = train_test_split(train_x,
                                                        train_y,
                                                        train_size=0.8, random_state=1234)


# In[49]:


xgb3 = XGBRegressor(n_estimators=500, learning_rate=0.05, seed=1234)


# In[50]:


tester = xgb3.fit(xtrain_x, xtrain_y, early_stopping_rounds=25, eval_metric='rmse', eval_set=[(xtest_x, xtest_y)])


# In[51]:


plt.plot(tester.evals_result()['validation_0']['rmse'][100:])
fig = plt.gcf()
fig.set_size_inches(12, 8)


# In[52]:


xgb3.score(test_x, test_y)


# In[69]:


plot_importance(xgb3)
fig = plt.gcf()
fig.set_size_inches(14,12)


# In[54]:


rmse(test_y, xgb3.predict(test_x), log_price=True)


# In[55]:


error_dict['Simple XGBoost with land ratios'] = rmse(test_y, xgb3.predict(test_x), log_price=True)


# In[68]:


plt.bar(range(len(error_dict)), list(error_dict.values()), align='center')
plt.xticks(range(len(error_dict)), list(error_dict.keys()), rotation=45)
fig = plt.gcf()
fig.set_size_inches(15,8)
fig.suptitle('Average error for each model')
plt.xlabel('Model type')
plt.ylabel('Error (RMSE)')


# In[57]:


error_dict


# ## Ensembling

# In[58]:


xgb_preds = xgb3.predict(test_x)
bgr_preds = bgr2.predict(nl(test_x))
lreg_preds = lreg2.predict(nl(test_x))


# In[59]:


all_preds = pd.DataFrame({'xgb': xgb_preds, 'bgr': bgr_preds, 'lreg': lreg_preds})


# In[60]:


ensemble_preds = (np.exp(all_preds.xgb) + np.exp(all_preds.bgr) + np.exp(all_preds.lreg)) / 3


# In[61]:


ensemble_preds.plot.hist(bins=50)


# In[62]:


np.exp(test_y).hist(bins=50)


# In[63]:


np.sqrt(np.mean((test_y.reset_index(drop=True) - ensemble_preds) ** 2))


# In[64]:


rmse(test_y.reset_index(drop=True), (all_preds.xgb + all_preds.bgr + all_preds.lreg) / 3, log_price=True)


# In[65]:


rmse(test_y.reset_index(drop=True), all_preds.xgb, log_price=True)


# In[ ]:




