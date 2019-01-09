
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
from datetime import datetime
import calendar
from math import sin, cos, sqrt, atan2, radians,asin
import folium
from folium import FeatureGroup, LayerControl, Map, Marker
from folium.plugins import HeatMap
from folium.plugins import TimestampedGeoJson
from folium.plugins import MarkerCluster
from geopy.distance import great_circle
import matplotlib.dates as mdates
import matplotlib as mpl
from datetime import timedelta
import datetime as dt
warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', -1)
plt.style.use('fivethirtyeight')
import folium
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import pickle
from geopy.distance import geodesic

df=pd.read_csv('finaldata1.csv')

df.ObservedDate.unique()
train = df[(df['ObservedTimestamp'] > '2017-07-22 07:00:00') & (df['ObservedTimestamp'] <= '2017-07-22 11:00:00') |                     (df['ObservedTimestamp'] > '2017-07-21 07:00:00') & (df['ObservedTimestamp'] <= '2017-07-21 11:00:00') |
(df['ObservedTimestamp'] > '2017-07-20 07:00:00') & (df['ObservedTimestamp'] <= '2017-07-20 11:00:00') |
(df['ObservedTimestamp'] > '2017-07-19 07:00:00') & (df['ObservedTimestamp'] <= '2017-07-19 11:00:00') |
(df['ObservedTimestamp'] > '2017-07-18 07:00:00') & (df['ObservedTimestamp'] <= '2017-07-18 11:00:00') |
(df['ObservedTimestamp'] > '2017-07-17 07:00:00') & (df['ObservedTimestamp'] <= '2017-07-17 11:00:00') |
(df['ObservedTimestamp'] > '2017-07-16 07:00:00') & (df['ObservedTimestamp'] <= '2017-07-16 11:00:00') |
(df['ObservedTimestamp'] > '2017-07-15 07:00:00') & (df['ObservedTimestamp'] <= '2017-07-15 11:00:00') |
(df['ObservedTimestamp'] > '2017-07-14 07:00:00') & (df['ObservedTimestamp'] <= '2017-07-14 11:00:00') |
(df['ObservedTimestamp'] > '2017-07-13 07:00:00') & (df['ObservedTimestamp'] <= '2017-07-13 11:00:00') |
(df['ObservedTimestamp'] > '2017-07-12 07:00:00') & (df['ObservedTimestamp'] <= '2017-07-12 11:00:00') |
(df['ObservedTimestamp'] > '2017-07-11 07:00:00') & (df['ObservedTimestamp'] <= '2017-07-11 11:00:00') |
(df['ObservedTimestamp'] > '2017-07-10 07:00:00') & (df['ObservedTimestamp'] <= '2017-07-10 11:00:00') |
(df['ObservedTimestamp'] > '2017-07-09 07:00:00') & (df['ObservedTimestamp'] <= '2017-07-09 11:00:00') |
(df['ObservedTimestamp'] > '2017-07-08 07:00:00') & (df['ObservedTimestamp'] <= '2017-07-08 11:00:00') |
(df['ObservedTimestamp'] > '2017-07-07 07:00:00') & (df['ObservedTimestamp'] <= '2017-07-07 11:00:00') |
(df['ObservedTimestamp'] > '2017-07-06 07:00:00') & (df['ObservedTimestamp'] <= '2017-07-06 11:00:00') |
(df['ObservedTimestamp'] > '2017-07-05 07:00:00') & (df['ObservedTimestamp'] <= '2017-07-05 11:00:00') |
(df['ObservedTimestamp'] > '2017-07-04 07:00:00') & (df['ObservedTimestamp'] <= '2017-07-04 11:00:00') |
(df['ObservedTimestamp'] > '2017-07-03 07:00:00') & (df['ObservedTimestamp'] <= '2017-07-03 11:00:00') |
(df['ObservedTimestamp'] > '2017-07-02 07:00:00') & (df['ObservedTimestamp'] <= '2017-07-02 11:00:00') |
(df['ObservedTimestamp'] > '2017-07-01 07:00:00') & (df['ObservedTimestamp'] <= '2017-07-01 11:00:00') ]  

train.isnull().sum()
train = train.dropna()
train.head()
train.Airline.unique()

#import csv as csv
#train.to_csv('peak.csv')


train.info()
train.Airline.unique()
train['ObservedTimestamp']=pd.to_datetime(train['ObservedTimestamp'],format='%Y-%m-%d %H:%M:%S')
train.head()


'''

Create datetime features based on pickup_datetime


'''

train['pickup_date']= train['ObservedTimestamp'].dt.date
train['pickup_day']=train['ObservedTimestamp'].apply(lambda x:x.day)
train['pickup_hour']=train['ObservedTimestamp'].apply(lambda x:x.hour)
train['pickup_day_of_week']=train['ObservedTimestamp'].apply(lambda x:calendar.day_name[x.weekday()])
train['pickup_month']=train['ObservedTimestamp'].apply(lambda x:x.month)
train['pickup_year']=train['ObservedTimestamp'].apply(lambda x:x.year)


train.isnull().sum()



'''

Exploratory Data Analysis
Distribution of Trip Fare

'''

plt.figure(figsize=(8,5))
sns.kdeplot(train['Fare']).set_title("Distribution of Trip Fare")


train.loc[train['Fare']<0].shape

train=train.loc[train['Fare']>=0]
train.shape


plt.figure(figsize=(8,5))
sns.kdeplot(np.log(train['Fare'].values)).set_title("Distribution of fare amount (log scale)")


#Distribution of Pickup and Dropoff Lat Lng

print("Range of Pickup Latitude is ", (min(train['pickup_latitude']),max(train['pickup_latitude'])))

print("Range of Dropoff Latitude is ", (min(train['dropoff_latitude']),max(train['dropoff_longitude'])))

#reading test data for analysis

test =  pd.read_csv('test.csv')
print("Longitude Boundary in test data")
min(test.pickup_longitude.min(), test.dropoff_longitude.min()),max(test.pickup_longitude.max(), test.dropoff_longitude.max())



boundary={'min_lng':70.263242,
              'min_lat':10.573143,
              'max_lng':78.986532, 
              'max_lat':20.709555}



train[(train.pickup_latitude==0) | (train.pickup_longitude)==0 | (train.dropoff_latitude==0)|(train.dropoff_longitude==0)].shape


train.loc[~((train.pickup_longitude >= boundary['min_lng'] ) & (train.pickup_longitude <= boundary['max_lng']) &
            (train.pickup_latitude >= boundary['min_lat']) & (train.pickup_latitude <= boundary['max_lat']) &
            (train.dropoff_longitude >= boundary['min_lng']) & (train.dropoff_longitude <= boundary['max_lng']) &
            (train.dropoff_latitude >=boundary['min_lat']) & (train.dropoff_latitude <= boundary['max_lat'])),'is_outlier_loc']=1
train.loc[((train.pickup_longitude >= boundary['min_lng'] ) & (train.pickup_longitude <= boundary['max_lng']) &
            (train.pickup_latitude >= boundary['min_lat']) & (train.pickup_latitude <= boundary['max_lat']) &
            (train.dropoff_longitude >= boundary['min_lng']) & (train.dropoff_longitude <= boundary['max_lng']) &
            (train.dropoff_latitude >=boundary['min_lat']) & (train.dropoff_latitude <= boundary['max_lat'])),'is_outlier_loc']=0

print("Outlier vs Non Outlier Counts")
print(train['is_outlier_loc'].value_counts())

# Let us drop rows, where location is outlier
train=train.loc[train['is_outlier_loc']==0]
train.drop(['is_outlier_loc'],axis=1,inplace=True)

train.shape

#Plot Heatmap of Pickups and Dropoffs 

#city_long_border = (72.85, 77.58)
#city_lat_border = (12.09, 19.97)
#
#train.plot(kind='scatter', x='dropoff_longitude', y='dropoff_latitude',
#                color='green', 
#                s=.02, alpha=.6)
#plt.title("Dropoffs")
#
#plt.ylim(city_lat_border)
#plt.xlim(city_long_border)
#
#
#train.plot(kind='scatter', x='pickup_longitude', y='pickup_latitude',
#                color='blue', 
#                s=.02, alpha=.6)
#plt.title("Pickups")
#
#plt.ylim(city_lat_border)
#plt.xlim(city_long_border)

'''

Heatmap based on fare amount

'''

train['pickup_latitude_round3']=train['pickup_latitude'].apply(lambda x:round(x,3))
train['pickup_longitude_round3']=train['pickup_longitude'].apply(lambda x:round(x,3))
train['dropoff_latitude_round3']=train['dropoff_latitude'].apply(lambda x:round(x,3))
train['dropoff_longitude_round3']=train['dropoff_longitude'].apply(lambda x:round(x,3))


'''

Average fare calcualtion between two airports


'''

pickup_fare_amount=train.groupby(['pickup_latitude_round3','pickup_longitude_round3'])['Fare'].mean().reset_index().rename(columns={'Fare':'avg_fare'})
pickup_fare_amount.head()



''''

Number of pickups and dropoffs from mumbai

'''



MUM={'min_lng':72.8352,
     'min_lat':12.6195,
     'max_lng':78.7401, 
     'max_lat':20.6659}
MUM_center=[15.6437,75.7900]
# Get all pickups to MUM
MUM_data=train.loc[(train.pickup_latitude>=MUM['min_lat']) & (train.pickup_latitude<=MUM['max_lat'])]
MUM_data=MUM_data.loc[(train.pickup_longitude>=MUM['min_lng']) & (train.pickup_longitude<=MUM['max_lng'])]

print("Number of Trips with Pickups from MUM",MUM_data.shape[0])

MUM_dropoff=train.loc[(train.dropoff_latitude>=MUM['min_lat']) & (train.dropoff_latitude<=MUM['max_lat'])]
MUM_dropoff=MUM_dropoff.loc[(train.dropoff_longitude>=MUM['min_lng']) & (train.dropoff_longitude<=MUM['max_lng'])]

print("Number of Trips with Dropoffs to BAN",MUM_dropoff.shape[0])

'''
# Create a folium map with MUM as the center 
m=folium.Map(location =MUM_center,zoom_start = 10,)
folium.Marker(location=MUM_center, popup='MUM Airport',icon=folium.Icon(color='black')).add_to(m)

mc = MarkerCluster().add_to(m)
#Add markers in blue for each pickup location and line between MUM and Pickup location over time. The thickness of line indicates the fare_amount

for index,row in MUM_data.iterrows():
    folium.Marker([row['dropoff_latitude'],row['dropoff_longitude']]).add_to(m)

'''

#Average Fare amount of trips from MUM
plt.figure(figsize=(8,5))
sns.kdeplot(np.log(MUM_data['Fare'].values),label='MUM Pickups')
sns.kdeplot(np.log(MUM_dropoff['Fare'].values),label='BAN Dropoff')
#sns.kdeplot(np.log(train['Fare'].values),label='All Trips in Train data')
plt.title("Fare Amount Distribution")



plt.figure(figsize=(8,5))
sns.kdeplot(np.log(MUM_dropoff['Fare'].values),label='MUM')
sns.kdeplot(np.log(train['Fare'].values),label='FARE')
plt.title("Dropoffs vs Fare Amount")




del MUM_data
del MUM
del MUM_dropoff


## Based on the above, let us create a function to see whether pickup or dropoff is an Airport. 

'''
def isAirport(latitude,longitude,airport_name='MUM'):
    if airport_name=='MUM':
        boundary={'min_lng':-73.8352,
     'min_lat':40.6195,
     'max_lng':-73.7401, 
     'max_lat':40.6659}
    elif airport_name=='EWR':
        boundary={
            'min_lng':-74.1925,
            'min_lat':40.6700, 
            'max_lng':-74.1531, 
            'max_lat':40.7081

        }
    elif airport_name=='la guardia':
        boundary={'min_lng':-73.8895, 
                  'min_lat':40.7664, 
                  'max_lng':-73.8550, 
                  'max_lat':40.7931
                 }
    if latitude>=boundary['min_lat'] and latitude<=boundary['max_lat']:
        if longitude>=boundary['min_lng'] and longitude<=boundary['max_lng']:
            return 1
    else:
        return 0
        


'''



'''

Calculating the total distance travelled by flight from Airport pickup to Airport Drop


'''

IND_airports={'MUM':{'min_lng':72.8352,
     'min_lat':12.6195,
     'max_lng':75.7401, 
     'max_lat':20.6659},
              
    'BAN':{'min_lng':75.1925,
            'min_lat':10.6700, 
            'max_lng':77.1531, 
            'max_lat':13.7081

        },
#    'LG':{'min_lng':75.8895, 
#                  'min_lat':10.7664, 
#                  'max_lng':77.8550, 
#                  'max_lat':13.7931
#        
#    }
    
}
    
    
def isAirport(latitude,longitude,airport_name='MUM'):
    
    if latitude>=IND_airports[airport_name]['min_lat'] and latitude<=IND_airports[airport_name]['max_lat'] and longitude>=IND_airports[airport_name]['min_lng'] and longitude<=IND_airports[airport_name]['max_lng']:
        return 1
    else:
        return 0
        


train['is_pickup_MUM']=train.apply(lambda row:isAirport(row['pickup_latitude'],row['pickup_longitude'],'MUM'),axis=1)
train['is_dropoff_MUM']=train.apply(lambda row:isAirport(row['dropoff_latitude'],row['dropoff_longitude'],'MUM'),axis=1)


train['is_pickup_BAN']=train.apply(lambda row:isAirport(row['pickup_latitude'],row['pickup_longitude'],'BAN'),axis=1)
train['is_dropoff_BAN']=train.apply(lambda row:isAirport(row['dropoff_latitude'],row['dropoff_longitude'],'BAN'),axis=1)


#calculate trip distance in miles
def distance(lat1, lat2, lon1,lon2):
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))



train['trip_distance']=train.apply(lambda row:distance(row['pickup_latitude'],row['dropoff_latitude'],row['pickup_longitude'],row['dropoff_longitude']),axis=1)


sns.kdeplot(np.log(train['trip_distance'].values)).set_title("Distribution of Trip Distance (log scale)")


plt.scatter(x=train['trip_distance'],y=train['Fare'])
plt.xlabel("Trip Distance")
plt.ylabel("Fare Amount")
plt.title("Trip Distance vs Fare Amount")

print("the distance travelled from MUMbai to BANgalore in miles is  :",train.trip_distance)

#
#
#non_airport=train.loc[(train['is_dropoff_MUM']==0) & (train['is_dropoff_BAN']==0) & (train['is_dropoff_LG']==0)]
#non_airport=non_airport.loc[(non_airport['is_pickup_MUM']==0) & (non_airport['is_pickup_BAN']==0) & (non_airport['is_pickup_LG']==0)]
#non_airport.shape
#
#
#
#plt.scatter(x=non_airport['trip_distance'],y=non_airport['Fare'])
#plt.xlabel("Trip Distance")
#plt.ylabel("Fare Amount")
#plt.title("Trip Distance vs Fare Amount (excluding airport)")
#
#
#
#non_airport_long_trips=non_airport[non_airport['trip_distance']>=50]
#
#
#
#
#drop_map = folium.Map(location = [40.730610,73.935242],zoom_start = 12,)
##print(pickup.shape)
#### For each pickup point add a circlemarker
#
#for index, row in non_airport_long_trips.iterrows():
#    
#    folium.CircleMarker([row['dropoff_latitude_round3'], row['dropoff_longitude_round3']],
#                        radius=3,
#                        
#                        color="#008000", 
#                        fill_opacity=0.9
#                       ).add_to(drop_map)
#for index, row in non_airport_long_trips.iterrows():
#    
#    folium.CircleMarker([row['pickup_latitude_round3'], row['pickup_longitude_round3']],
#                        radius=3,
#                        
#                        color="blue", 
#                        fill_opacity=0.9
#                       ).add_to(drop_map)
#
#'''
#hm_wide = HeatMap( list(zip(drop.dropoff_latitude_round3.values, drop.dropoff_longitude_round3.values, drop.Num_Trips.values)),
#                     min_opacity=0.2,
#                     radius=5, blur=15,
#                     max_zoom=1 
#                 )
#drop_map.add_child(hm_wide)
#
#'''
#
#drop_map
#


'''

Modelling the data

'''




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import calendar
import warnings
from math import sin, cos, sqrt, atan2, radians,asin

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
warnings.filterwarnings('ignore')


#train=pd.read_csv("finaldata1.csv")
print("Shape of Training Data",train.shape)
test=pd.read_csv("test.csv")
print("Shape of Testing Data", test.shape)



'''

drop row if fare < 0

'''

#
#def encodeDays(day_of_week):
#    day_dict={'Sunday':0,'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6}
#    return day_dict[day_of_week]
#def clean_data(data):
#    boundary={'min_lng':74.263242,
#              'min_lat':10.573143,
#              'max_lng':79.986532, 
#              'max_lat':20.709555}
#    
#    data['pickup_datetime']=pd.to_datetime(data['ObservedTimestamp'],format='%Y-%m-%d %H:%M:%S')
#    data['pickup_day']=data['ObservedTimestamp'].apply(lambda x:x.day)
#    data['pickup_hour']=data['ObservedTimestamp'].apply(lambda x:x.hour)
#    data['pickup_day_of_week']=data['ObservedTimestamp'].apply(lambda x:calendar.day_name[x.weekday()])
#    data['pickup_month']=data['ObservedTimestamp'].apply(lambda x:x.month)
#    data['pickup_year']=data['ObservedTimestamp'].apply(lambda x:x.year)
#    if 'Fare' in data.columns:
#        data=data[data['Fare']>=0]
#        data.loc[~((data.pickup_longitude >= boundary['min_lng'] ) & (data.pickup_longitude <= boundary['max_lng']) &
#            (data.pickup_latitude >= boundary['min_lat']) & (data.pickup_latitude <= boundary['max_lat']) &
#            (data.dropoff_longitude >= boundary['min_lng']) & (data.dropoff_longitude <= boundary['max_lng']) &
#            (data.dropoff_latitude >=boundary['min_lat']) & (data.dropoff_latitude <= boundary['max_lat'])),'is_outlier_loc']=1
#        data.loc[((data.pickup_longitude >= boundary['min_lng'] ) & (data.pickup_longitude <= boundary['max_lng']) &
#            (data.pickup_latitude >= boundary['min_lat']) & (data.pickup_latitude <= boundary['max_lat']) &
#            (data.dropoff_longitude >= boundary['min_lng']) & (data.dropoff_longitude <= boundary['max_lng']) &
#            (data.dropoff_latitude >=boundary['min_lat']) & (data.dropoff_latitude <= boundary['max_lat'])),'is_outlier_loc']=0
#
#    #print("Outlier vs Non Outlier Counts")
#    #print(data['is_outlier_loc'].value_counts())
#
#    # Let us drop rows, where location is outlier
#        data=data.loc[data['is_outlier_loc']==0]
#        data.drop(['is_outlier_loc'],axis=1,inplace=True)
#    
##    data=data[data['passenger_count']<=8]
#    data['pickup_day_of_week']=data['pickup_day_of_week'].apply(lambda x:encodeDays(x))
#    return data
#
#
#
#train=clean_data(train)
##train.drop(['ObservedTimestamp','ObservedDate','ObservedTime','pickup_datetime'],axis=1)
#test=clean_data(test)
#print("Shape of Training Data after cleaning ",train.shape)
#print("Shape of Testing Data after cleaning", test.shape)
#


'''

Dropping unwanted columns from the data
One Hot Encoding of categorical variables
Dividing training data into train and validation data sets
features and target varible must be seperated
split ratio must be passed as an argument



'''

def processDataForModelling(data,target,drop_cols,is_train=True,split=0.25):
    data_1=data.drop(drop_cols,axis=1)
    # One hot Encoding
    data_1=pd.get_dummies(data_1)
    if is_train==True:
        X=data_1.drop([target],axis=1)
        y=data_1[target]
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=split,random_state=123)
        
        print("Shape of Training Features",X_train.shape)
        print("Shape of Validation Features ",X_test.shape)
        
        return X_train, X_test, y_train, y_test
    else:
        print ("Shape of Test Data",data_1.shape)
        return data_1



X_train, X_test, y_train, y_test=processDataForModelling(train,'Fare',drop_cols=['ObservedTimestamp'],is_train=True,split=0.2)

test_data=processDataForModelling(test,'Fare',drop_cols=['ObservedTimestamp'],is_train=False)

#X_train = X_train.drop(['ObservedTimestamp'],axis=1)
#X_test = X_test.drop(['ObservedTimestamp'],axis = 1)


'''

Building a Baseline Model and Identifying a good ML algorithm for this problem

'''

avg_fare=round(np.mean(y_train),2)
avg_fare


baseline_pred=np.repeat(avg_fare,y_test.shape[0])
baseline_rmse=np.sqrt(mean_squared_error(baseline_pred, y_test))
print("Basline RMSE of Validation data :",baseline_rmse)



#X_train['ObservedTimestamp'] = pd.to_numeric(df['ObservedTimestamp'],errors='coerce')

'''
Building Linear Regression Model
'''

lm = LinearRegression()
lm.fit(X_train,y_train)
y_pred=np.round(lm.predict(X_test),2)
lm_rmse=np.sqrt(mean_squared_error(y_pred, y_test))
print("RMSE for Linear Regression is ",lm_rmse)

'''
Building Random Forest Model
'''

rf = RandomForestRegressor(n_estimators = 100, random_state = 883,n_jobs=-1)
rf.fit(X_train,y_train)

rf_pred= rf.predict(X_test)
rf_rmse=np.sqrt(mean_squared_error(rf_pred, y_test))
print("RMSE for Random Forest is ",rf_rmse)

'''
Building LightGBM algorithm
'''
train_data=lgb.Dataset(X_train,label=y_train)

param = {'num_leaves':31, 'num_trees':5000, 'objective':'regression'}
param['metric'] = 'l2_root'


num_round=5000
cv_results = lgb.cv(param, train_data, num_boost_round=num_round, nfold=10,verbose_eval=20, early_stopping_rounds=20,stratified=False)


print('Best num_boost_round:', len(cv_results['rmse-mean']))
#lgb_pred = lgb_bst.predict(X_test)

lgb_bst=lgb.train(param,train_data,len(cv_results['rmse-mean']))


lgb_pred = lgb_bst.predict(X_test)
lgb_rmse=np.sqrt(mean_squared_error(lgb_pred, y_test))
print("RMSE for Light GBM is ",lgb_rmse)



dtrain = xgb.DMatrix(X_train,label=y_train)
dtest = xgb.DMatrix(X_test,label=y_test)


xgb_param = {'objective':'reg:linear','eval_metric':'rmse'}
xgb_cv=xgb.cv(xgb_param, dtrain, num_boost_round=5000, nfold=5,early_stopping_rounds=20)


nrounds=xgb_cv.shape[0]

xbg_model=xgb.train(params={'objective':'reg:linear','eval_metric':'rmse'}
                    ,dtrain=dtrain,num_boost_round=nrounds)




xgb_pred=xbg_model.predict(dtest)
xgb_rmse=np.sqrt(mean_squared_error(xgb_pred, y_test))
print("RMSE for XGBOOST is ",xgb_rmse)


'''

 train RMSE and test RMSE - This will help us understand variance in the model and get the best model

'''

model_pred=pd.DataFrame()
model_pred['model_name']=['Linear Regression','Random Forest','Light GBM','XGBOOST']
model_pred['test_rmse']=[lm_rmse,rf_rmse,lgb_rmse,xgb_rmse]


lm_train_rmse=np.sqrt(mean_squared_error(lm.predict(X_train), y_train))
rf_train_rmse=np.sqrt(mean_squared_error(rf.predict(X_train),y_train))
lgb_train_rmse=np.sqrt(mean_squared_error(lgb_bst.predict(X_train),y_train))
xgb_train_rmse=np.sqrt(mean_squared_error(xbg_model.predict(dtrain),y_train))


model_pred['train_rmse']=[lm_train_rmse,rf_train_rmse,lgb_train_rmse,xgb_train_rmse]
model_pred['variance']=model_pred['train_rmse'] - model_pred['test_rmse']
model_pred

'''

Feature Engineering

'''

IND_airports={'MUM':{'min_lng':73.8352,
     'min_lat':40.6195,
     'max_lng':73.7401, 
     'max_lat':40.6659},
              
    'BAN':{'min_lng':73.1925,
            'min_lat':10.6700, 
            'max_lng':79.1531, 
            'max_lat':20.7081

        },
    'LG':{'min_lng':73.8895, 
                  'min_lat':40.7664, 
                  'max_lng':73.8550, 
                  'max_lat':40.7931
    }    
    
    
}
    
def isAirport(latitude,longitude,airport_name='MUM'):
    
    if latitude>=IND_airports[airport_name]['min_lat'] and latitude<=IND_airports[airport_name]['max_lat'] and longitude>=IND_airports[airport_name]['min_lng'] and longitude<=IND_airports[airport_name]['max_lng']:
        return 1
    else:
        return 0


IND_boroughs={
    'man':{
        'min_lng':74.0479,
        'min_lat':40.6829,
        'max_lng':73.9067,
        'max_lat':40.8820
    },
    
    'qu':{
        'min_lng':73.9630,
        'min_lat':40.5431,
        'max_lng':73.7004,
        'max_lat':40.8007

    },

    'brn':{
        'min_lng':74.0421,
        'min_lat':40.5707,
        'max_lng':73.8334,
        'max_lat':40.7395

    },

    'bro':{
        'min_lng':73.9339,
        'min_lat':40.7855,
        'max_lng':73.7654,
        'max_lat':40.9176

    },

    'stand':{
        'min_lng':74.2558,
        'min_lat':40.4960,
        'max_lng':74.0522,
        'max_lat':40.6490

    }
}


def isAirport(latitude,longitude,airport_name='MUM'):
    
    if latitude>=IND_airports[airport_name]['min_lat'] and latitude<=IND_airports[airport_name]['max_lat'] and longitude>=IND_airports[airport_name]['min_lng'] and longitude<=IND_airports[airport_name]['max_lng']:
        return 1
    else:
        return 0



X_train['is_pickup_LG']=X_train.apply(lambda row:isAirport(row['pickup_latitude'],row['pickup_longitude'],'LG'),axis=1)
X_train['is_dropoff_LG']=X_train.apply(lambda row:isAirport(row['dropoff_latitude'],row['dropoff_longitude'],'LG'),axis=1)
X_train['is_pickup_BAN']=X_train.apply(lambda row:isAirport(row['pickup_latitude'],row['pickup_longitude'],'BAN'),axis=1)
X_train['is_dropoff_BAN']=X_train.apply(lambda row:isAirport(row['dropoff_latitude'],row['dropoff_longitude'],'BAN'),axis=1)
X_train['is_pickup_MUM']=X_train.apply(lambda row:isAirport(row['pickup_latitude'],row['pickup_longitude'],'MUM'),axis=1)
X_train['is_dropoff_MUM']=X_train.apply(lambda row:isAirport(row['dropoff_latitude'],row['dropoff_longitude'],'MUM'),axis=1)




X_test['is_pickup_LG']=X_test.apply(lambda row:isAirport(row['pickup_latitude'],row['pickup_longitude'],'LG'),axis=1)
X_test['is_dropoff_LG']=X_test.apply(lambda row:isAirport(row['dropoff_latitude'],row['dropoff_longitude'],'LG'),axis=1)
X_test['is_pickup_BAN']=X_test.apply(lambda row:isAirport(row['pickup_latitude'],row['pickup_longitude'],'BAN'),axis=1)
X_test['is_dropoff_BAN']=X_test.apply(lambda row:isAirport(row['dropoff_latitude'],row['dropoff_longitude'],'BAN'),axis=1)
X_test['is_pickup_MUM']=X_test.apply(lambda row:isAirport(row['pickup_latitude'],row['pickup_longitude'],'MUM'),axis=1)
X_test['is_dropoff_MUM']=X_test.apply(lambda row:isAirport(row['dropoff_latitude'],row['dropoff_longitude'],'MUM'),axis=1)


test['is_pickup_LG']=test.apply(lambda row:isAirport(row['pickup_latitude'],row['pickup_longitude'],'LG'),axis=1)
test['is_dropoff_LG']=test.apply(lambda row:isAirport(row['dropoff_latitude'],row['dropoff_longitude'],'LG'),axis=1)
test['is_pickup_BAN']=test.apply(lambda row:isAirport(row['pickup_latitude'],row['pickup_longitude'],'BAN'),axis=1)
test['is_dropoff_BAN']=test.apply(lambda row:isAirport(row['dropoff_latitude'],row['dropoff_longitude'],'BAN'),axis=1)
test['is_pickup_MUM']=test.apply(lambda row:isAirport(row['pickup_latitude'],row['pickup_longitude'],'MUM'),axis=1)
test['is_dropoff_MUM']=test.apply(lambda row:isAirport(row['dropoff_latitude'],row['dropoff_longitude'],'MUM'),axis=1)



def getBorough(lat,lng):
    
    locs=IND_boroughs.keys()
    for loc in locs:
        if lat>=IND_boroughs[loc]['min_lat'] and lat<=IND_boroughs[loc]['max_lat'] and lng>=IND_boroughs[loc]['min_lng'] and lng<=IND_boroughs[loc]['max_lng']:
            return loc
    return 'others'
X_train['pickup_borough']=X_train.apply(lambda row:getBorough(row['pickup_latitude'],row['pickup_longitude']),axis=1)
X_train['dropoff_borough']=X_train.apply(lambda row:getBorough(row['dropoff_latitude'],row['dropoff_longitude']),axis=1)
X_test['pickup_borough']=X_test.apply(lambda row:getBorough(row['pickup_latitude'],row['pickup_longitude']),axis=1)
X_test['dropoff_borough']=X_test.apply(lambda row:getBorough(row['dropoff_latitude'],row['dropoff_longitude']),axis=1)

test['pickup_borough']=test.apply(lambda row:getBorough(row['pickup_latitude'],row['pickup_longitude']),axis=1)
test['dropoff_borough']=test.apply(lambda row:getBorough(row['dropoff_latitude'],row['dropoff_longitude']),axis=1)

X_train=pd.get_dummies(X_train)
X_test=pd.get_dummies(X_test)
test=pd.get_dummies(test)


lower_manhattan_boundary={'min_lng':71.0194,
                          'min_lat':10.6997,
                          'max_lng':79.9716,
                          'max_lat':20.7427}



def isLowerManhattan(lat,lng):
    if lat>=lower_manhattan_boundary['min_lat'] and lat<=lower_manhattan_boundary['max_lat'] and lng>=lower_manhattan_boundary['min_lng'] and lng<=lower_manhattan_boundary['max_lng']:
        return 1
    else:
        return 0


X_train['is_pickup_lower_manhattan']=X_train.apply(lambda row:isLowerManhattan(row['pickup_latitude'],row['pickup_longitude']),axis=1)
X_train['is_dropoff_lower_manhattan']=X_train.apply(lambda row:isLowerManhattan(row['dropoff_latitude'],row['dropoff_longitude']),axis=1)


X_test['is_pickup_lower_manhattan']=X_test.apply(lambda row:isLowerManhattan(row['pickup_latitude'],row['pickup_longitude']),axis=1)
X_test['is_dropoff_lower_manhattan']=X_test.apply(lambda row:isLowerManhattan(row['dropoff_latitude'],row['dropoff_longitude']),axis=1)

test['is_pickup_lower_manhattan']=test.apply(lambda row:isLowerManhattan(row['pickup_latitude'],row['pickup_longitude']),axis=1)
test['is_dropoff_lower_manhattan']=test.apply(lambda row:isLowerManhattan(row['dropoff_latitude'],row['dropoff_longitude']),axis=1)



def distance(lat1,lon1,lat2,lon2):
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))
#X_train['trip_distance']=X_train.apply(lambda row:distance(row['pickup_latitude'],row['dropoff_latitude'],row['pickup_longitude'],row['dropoff_longitude']),axis=1)
#X_test['trip_distance']=X_test.apply(lambda row:distance(row['pickup_latitude'],row['dropoff_latitude'],row['pickup_longitude'],row['dropoff_longitude']),axis=1)

#test['trip_distance']=test.apply(lambda row:distance(row['pickup_latitude'],row['dropoff_latitude'],row['pickup_longitude'],row['dropoff_longitude']),axis=1)

lgr=(73.8733, 40.7746)
MUM=(73.7900, 40.6437)
ewr=(74.1843, 40.6924)

test['pickup_distance_MUM']=test.apply(lambda row:distance(row['pickup_latitude'],row['pickup_longitude'],MUM[1],MUM[0]),axis=1)
test['dropoff_distance_MUM']=test.apply(lambda row:distance(row['dropoff_latitude'],row['dropoff_longitude'],MUM[1],MUM[0]),axis=1)
test['pickup_distance_ewr']=test.apply(lambda row:distance(row['pickup_latitude'],row['pickup_longitude'],ewr[1],ewr[0]),axis=1)
test['dropoff_distance_ewr']=test.apply(lambda row:distance(row['dropoff_latitude'],row['dropoff_longitude'],ewr[1],ewr[0]),axis=1)
test['pickup_distance_LG']=test.apply(lambda row:distance(row['pickup_latitude'],row['pickup_longitude'],lgr[1],lgr[0]),axis=1)
test['dropoff_distance_LG']=test.apply(lambda row:distance(row['dropoff_latitude'],row['dropoff_longitude'],lgr[1],lgr[0]),axis=1)

X_train['pickup_distance_MUM']=X_train.apply(lambda row:distance(row['pickup_latitude'],row['pickup_longitude'],MUM[1],MUM[0]),axis=1)
X_train['dropoff_distance_MUM']=X_train.apply(lambda row:distance(row['dropoff_latitude'],row['dropoff_longitude'],MUM[1],MUM[0]),axis=1)
X_train['pickup_distance_ewr']=X_train.apply(lambda row:distance(row['pickup_latitude'],row['pickup_longitude'],ewr[1],ewr[0]),axis=1)
X_train['dropoff_distance_ewr']=X_train.apply(lambda row:distance(row['dropoff_latitude'],row['dropoff_longitude'],ewr[1],ewr[0]),axis=1)
X_train['pickup_distance_LG']=X_train.apply(lambda row:distance(row['pickup_latitude'],row['pickup_longitude'],lgr[1],lgr[0]),axis=1)
X_train['dropoff_distance_LG']=X_train.apply(lambda row:distance(row['dropoff_latitude'],row['dropoff_longitude'],lgr[1],lgr[0]),axis=1)

X_test['pickup_distance_MUM']=X_test.apply(lambda row:distance(row['pickup_latitude'],row['pickup_longitude'],MUM[1],MUM[0]),axis=1)
X_test['dropoff_distance_MUM']=X_test.apply(lambda row:distance(row['dropoff_latitude'],row['dropoff_longitude'],MUM[1],MUM[0]),axis=1)
X_test['pickup_distance_ewr']=X_test.apply(lambda row:distance(row['pickup_latitude'],row['pickup_longitude'],ewr[1],ewr[0]),axis=1)
X_test['dropoff_distance_ewr']=X_test.apply(lambda row:distance(row['dropoff_latitude'],row['dropoff_longitude'],ewr[1],ewr[0]),axis=1)
X_test['pickup_distance_LG']=X_test.apply(lambda row:distance(row['pickup_latitude'],row['pickup_longitude'],lgr[1],lgr[0]),axis=1)
X_test['dropoff_distance_LG']=X_test.apply(lambda row:distance(row['dropoff_latitude'],row['dropoff_longitude'],lgr[1],lgr[0]),axis=1)




manhattan=(73.9664, 40.7909)
queens=(73.8317, 40.7038)
brooklyn=(73.9489, 40.6551)
bronx=(73.8568, 40.8572)
staten_island=(74.1540, 40.5725)




test['pickup_distance_manhattan']=test.apply(lambda row:distance(row['pickup_latitude'],row['pickup_longitude'],manhattan[1],manhattan[0]),axis=1)
test['pickup_distance_queens']=test.apply(lambda row:distance(row['pickup_latitude'],row['pickup_longitude'],queens[1],queens[0]),axis=1)
test['pickup_distance_brooklyn']=test.apply(lambda row:distance(row['pickup_latitude'],row['pickup_longitude'],brooklyn[1],brooklyn[0]),axis=1)
test['pickup_distance_bronx']=test.apply(lambda row:distance(row['pickup_latitude'],row['pickup_longitude'],bronx[1],bronx[0]),axis=1)
test['pickup_distance_statenisland']=test.apply(lambda row:distance(row['pickup_latitude'],row['pickup_longitude'],staten_island[1],staten_island[0]),axis=1)





test['dropoff_distance_manhattan']=test.apply(lambda row:distance(row['dropoff_latitude'],row['dropoff_longitude'],manhattan[1],manhattan[0]),axis=1)
test['dropoff_distance_queens']=test.apply(lambda row:distance(row['dropoff_latitude'],row['dropoff_longitude'],queens[1],queens[0]),axis=1)
test['dropoff_distance_brooklyn']=test.apply(lambda row:distance(row['dropoff_latitude'],row['dropoff_longitude'],brooklyn[1],brooklyn[0]),axis=1)
test['dropoff_distance_bronx']=test.apply(lambda row:distance(row['dropoff_latitude'],row['dropoff_longitude'],bronx[1],bronx[0]),axis=1)
test['dropoff_distance_statenisland']=test.apply(lambda row:distance(row['dropoff_latitude'],row['dropoff_longitude'],staten_island[1],staten_island[0]),axis=1)


X_train['pickup_distance_manhattan']=X_train.apply(lambda row:distance(row['pickup_latitude'],row['pickup_longitude'],manhattan[1],manhattan[0]),axis=1)
X_train['pickup_distance_queens']=X_train.apply(lambda row:distance(row['pickup_latitude'],row['pickup_longitude'],queens[1],queens[0]),axis=1)
X_train['pickup_distance_brooklyn']=X_train.apply(lambda row:distance(row['pickup_latitude'],row['pickup_longitude'],brooklyn[1],brooklyn[0]),axis=1)
X_train['pickup_distance_bronx']=X_train.apply(lambda row:distance(row['pickup_latitude'],row['pickup_longitude'],bronx[1],bronx[0]),axis=1)
X_train['pickup_distance_statenisland']=X_train.apply(lambda row:distance(row['pickup_latitude'],row['pickup_longitude'],staten_island[1],staten_island[0]),axis=1)

X_train['dropoff_distance_manhattan']=X_train.apply(lambda row:distance(row['dropoff_latitude'],row['dropoff_longitude'],manhattan[1],manhattan[0]),axis=1)
X_train['dropoff_distance_queens']=X_train.apply(lambda row:distance(row['dropoff_latitude'],row['dropoff_longitude'],queens[1],queens[0]),axis=1)
X_train['dropoff_distance_brooklyn']=X_train.apply(lambda row:distance(row['dropoff_latitude'],row['dropoff_longitude'],brooklyn[1],brooklyn[0]),axis=1)
X_train['dropoff_distance_bronx']=X_train.apply(lambda row:distance(row['dropoff_latitude'],row['dropoff_longitude'],bronx[1],bronx[0]),axis=1)
X_train['dropoff_distance_statenisland']=X_train.apply(lambda row:distance(row['dropoff_latitude'],row['dropoff_longitude'],staten_island[1],staten_island[0]),axis=1)




X_test['pickup_distance_manhattan']=X_test.apply(lambda row:distance(row['pickup_latitude'],row['pickup_longitude'],manhattan[1],manhattan[0]),axis=1)
X_test['pickup_distance_queens']=X_test.apply(lambda row:distance(row['pickup_latitude'],row['pickup_longitude'],queens[1],queens[0]),axis=1)
X_test['pickup_distance_brooklyn']=X_test.apply(lambda row:distance(row['pickup_latitude'],row['pickup_longitude'],brooklyn[1],brooklyn[0]),axis=1)
X_test['pickup_distance_bronx']=X_test.apply(lambda row:distance(row['pickup_latitude'],row['pickup_longitude'],bronx[1],bronx[0]),axis=1)
X_test['pickup_distance_statenisland']=X_test.apply(lambda row:distance(row['pickup_latitude'],row['pickup_longitude'],staten_island[1],staten_island[0]),axis=1)

X_test['dropoff_distance_manhattan']=X_test.apply(lambda row:distance(row['dropoff_latitude'],row['dropoff_longitude'],manhattan[1],manhattan[0]),axis=1)
X_test['dropoff_distance_queens']=X_test.apply(lambda row:distance(row['dropoff_latitude'],row['dropoff_longitude'],queens[1],queens[0]),axis=1)
X_test['dropoff_distance_brooklyn']=X_test.apply(lambda row:distance(row['dropoff_latitude'],row['dropoff_longitude'],brooklyn[1],brooklyn[0]),axis=1)
X_test['dropoff_distance_bronx']=X_test.apply(lambda row:distance(row['dropoff_latitude'],row['dropoff_longitude'],bronx[1],bronx[0]),axis=1)
X_test['dropoff_distance_statenisland']=X_test.apply(lambda row:distance(row['dropoff_latitude'],row['dropoff_longitude'],staten_island[1],staten_island[0]),axis=1)


X_train.to_csv("X_train_cleaned.csv",index=False)
X_test.to_csv("X_test_cleaned.csv",index=False)
test.to_csv("test_cleaned.csv",index=False)


train_data=lgb.Dataset(X_train,label=y_train)
param = {'num_leaves':31, 'num_trees':5000, 'objective':'regression'}
param['metric'] = 'l2_root'

num_round=5000
cv_results = lgb.cv(param, train_data, num_boost_round=num_round, nfold=5,verbose_eval=20, early_stopping_rounds=20,stratified=False)


print('Best num_boost_round:', len(cv_results['rmse-mean']))

lgb_bst=lgb.train(param,train_data,len(cv_results['rmse-mean']))

lgb_pred = lgb_bst.predict(X_test)
lgb_rmse=np.sqrt(mean_squared_error(lgb_pred, y_test))
print("RMSE for Light GBM with Feature Engineering is ",lgb_rmse)


lgb_train_rmse=np.sqrt(mean_squared_error(lgb_bst.predict(X_train),y_train))
print("Train RMSE for Light GBM with Feature Engineering is", lgb_train_rmse)


variance=lgb_train_rmse - lgb_rmse
print("Variance of Light GBM with Feature Engineering is ", variance)


'''

Tuning Light GBM

'''



param={'metric': 'l2_root',
 
 'objective': 'regression',
 'verbose': 1,
 #'num_trees':1000
      } #Light GBM with these params had an test rmse on 3.65


gridparams={
    'learning_rate':[0.1,0.75,0.005,0.025],
    #'num_iterations':[100,250,500],
    'num_leaves':[31,60],
    'bagging_freq':[10,20],
    'bagging_fraction':[0.85,1,0.9,0.95],
    'boosting_type':['gbdt'],
    #'subsample':[1,0.8,0.75],
    #'colsample_bytree':[0.8,0.7,1],
    'max_depth':[-1,6,5]
}


mdl = lgb.LGBMRegressor(
          objective = 'regression',
#          n_jobs = -1,
          n_jobs = 1, # Updated from 'nthread'
          verbose=1,
          metric='l2_root')


grid = GridSearchCV(mdl, gridparams,
                    verbose=1,
                    cv=3,
                    n_jobs=-1)


grid.fit(X_train,y_train)

grid.best_params_

bst_params={'metric': 'l2_root',
 'num_leaves': grid.best_params_['num_leaves'],
 
  'learning_rate':grid.best_params_['learning_rate'],
#  'n_estimators':grid.best_params_['n_estimators'],
            
 'objective': 'regression',
 'verbose': 1,
 }


num_round=5000
cv_results = lgb.cv(bst_params, train_data, num_boost_round=num_round, nfold=5,verbose_eval=20, early_stopping_rounds=20,stratified=False)


print('Best num_boost_round:', len(cv_results['rmse-mean']))

lgb_bst=lgb.train(bst_params,train_data)


lgb_pred = lgb_bst.predict(X_test)
lgb_rmse=np.sqrt(mean_squared_error(lgb_pred, y_test))
print("RMSE for Light GBM with Feature Engineering is ",lgb_rmse)
lgb_train_rmse=np.sqrt(mean_squared_error(lgb_bst.predict(X_train),y_train))
print("Train RMSE for Light GBM with Feature Engineering is", lgb_train_rmse)
variance=lgb_train_rmse - lgb_rmse
print("Variance of Light GBM with Feature Engineering is ", variance)




'''

XGB Regressor using Hyperopt and get the best params

'''
X_train=pd.read_csv("X_train_cleaned.csv")
X_test=pd.read_csv("X_test_cleaned.csv")
#y_train=pd.read_csv("Y_Train.csv")
#y_test=pd.read_csv("Y_test.csv")


def objective(space):

    clf = lgb.LGBMRegressor(
          objective = 'regression',
          n_jobs = -1, # Updated from 'nthread'
          verbose=1,
          boosting_type='gbdt',
        num_leaves=60,
        bagging_freq=20,
       subsample_freq=100,
    max_depth=int(space['max_depth']),
    subsample=space['subsample'],
        n_estimators=5000,
    colsample_bytree=space['colsample'])
          #metric='l2_root')

    eval_set=[( X_train, y_train), ( X_test,y_test)]

    clf.fit(X_train, np.array(y_train),
            eval_set=eval_set,eval_metric='rmse',
            early_stopping_rounds=20)

    pred = clf.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    print("SCORE:", rmse)

    return{'loss':rmse, 'status': STATUS_OK }



space ={
        'max_depth': hp.quniform("x_max_depth", 5, 30, 3),
        #'min_child_weight': hp.quniform ('x_min_child', 1, 10, 1),
        'subsample': hp.uniform ('x_subsample', 0.8, 1),
        'colsample':hp.uniform ('x_colsample', 0.3, 1)
    }



from hyperopt.mongoexp import MongoTrials
trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials)

print(best)




clf = lgb.LGBMRegressor(
          objective = 'regression',
          n_jobs = -1, # Updated from 'nthread'
          verbose=1,
          boosting_type='gbdt',
        num_leaves=60,
        bagging_freq=20,
       subsample_freq=100,
    max_depth=int(best['x_max_depth']),
    subsample=best['x_subsample'],
        n_estimators=5000,
    colsample_bytree=best['x_colsample'])
          #metric='l2_root')



eval_set=[( X_train, y_train), ( X_test,y_test)]
clf.fit(X_train, np.array(y_train),eval_set=eval_set,eval_metric='rmse',early_stopping_rounds=20)

valid_pred = clf.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, valid_pred))
print("Validation RMSE after tuning ",rmse)
pred = clf.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, pred))
print("Train RMSE after tuning",train_rmse)
print("Variance of model ", abs(train_rmse - rmse))


lgbm_feature_importance=sorted(zip(map(lambda x: round(x, 4),clf.feature_importances_), X_train.columns),reverse=True)
plt.figure(figsize=(10,15))
sns.barplot([imp for imp,feature in lgbm_feature_importance],[feature for imp,feature in lgbm_feature_importance])



sns.kdeplot(np.log(y_test),label='actual fare amount')
sns.kdeplot(np.log(valid_pred),label='predicted fare amount')













































'''








plt.figure(figsize=(8,5))
sns.kdeplot(np.log(train['Fare'].values),label='Arr Pickups')
#sns.kdeplot(np.log(MUM_dropoff['fare_amount'].values),label='MUM Dropoff')
sns.kdeplot(np.log(train['Fare'].values),label='All Trips in Train data')
plt.title("Fare Amount Distribution")


plt.figure(figsize=(8,5))
sns.kdeplot(np.log(train['Fare'].values),label='MUM')
sns.kdeplot(np.log(train['Fare'].values),label='train')
plt.title("Dropoffs vs Fare Amount")




df = df.set_index(df['ObservedTimestamp'])
df.loc['2017-07-17 07:00:00':'2017-07-18 11:00:00']


X = df.set_index('DateTime')
by_time = y.groupby(y.index.Fare)
hourly_ticks = 4 * 60 * 60 * np.arange(6)
by_time.plot(xticks=hourly_ticks, style=[':', '--', '-']);



#df['new_date'] = [d.date() for d in df['Observed Timestamp']]
#df['new_time'] = [d.time() for d in df['Observed Timestamp']]

useful_col = ['Observed Timestamp','Fare']
useful_col = ['Observed Time','Fare']
useful_col = ['Observed Date','Fare']


useful_col = ['Observed Date','Observed Time','Fare']
df.loc[:,useful_col].to_csv('cleandata1.csv')

from pandas import Series
from matplotlib import pyplot
series = pd.read_csv('cleandata1.csv',index_col=0)
print(series.head())
series.plot()
pyplot.show()


'''

#select time series forecast model


'''



from pandas import Series
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
import numpy
 
# create a difference transform of the dataset
def difference(dataset):
	diff = list()
	for i in range(1, len(dataset)):
		value = dataset[i] - dataset[i - 1]
		diff.append(value)
	return numpy.array(diff)
 
# Make a prediction give regression coefficients and lag obs
def predict(coef, history):
	yhat = coef[0]
	for i in range(1, len(coef)):
		yhat += coef[i] * history[-i]
	return yhat
 
#series = Series.from_csv('cleandata1.csv', header=0)
    
series = pd.read_csv('cleandata1.csv',header=None,index_col=0)

series.set_index('Observed Date',inplace=True).diff()
# split dataset
X = difference(series.values)
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:]
# train autoregression
model = AR(train)
model_fit = model.fit(maxlag=6, disp=False)
window = model_fit.k_ar
coef = model_fit.params
# walk forward over time steps in test
history = [train[i] for i in range(len(train))]
predictions = list()
for t in range(len(test)):
	yhat = predict(coef, history)
	obs = test[t]
	predictions.append(yhat)
	history.append(obs)
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()












flight_fare = df.loc[:,'Fare']

#furniture['Order Date'].min(), furniture['Order Date'].max()

df['Observed Timestamp'].min()
df['Observed Timestamp'].max()


#cols = ['Row ID', 'Order ID', 'Ship Date', 'Ship Mode', 'Customer ID', 'Customer Name', 'Segment', 'Country', 'City', 'State', 'Postal Code', 'Region', 'Product ID', 'Category', 'Sub-Category', 'Product Name', 'Quantity', 'Discount', 'Profit']
#furniture.drop(cols, axis=1, inplace=True)
#furniture = furniture.sort_values('Order Date')
#furniture.isnull().sum()

df_flights_new = df[['Observed Timestamp','Fare']]
#df_flights_new = df.loc[:,['Observed Timestamp','Fare']]

df_flights_new = df_flights_new.sort_values('Observed Timestamp')
df_flights_new.isnull().sum()


#furniture = furniture.groupby('Order Date')['Sales'].sum().reset_index()

###df_flights_new_sum =df_flights_new.groupby('Observed Timestamp')['Fare'].sum().reset_index()


#furniture = furniture.set_index('Order Date')
#furniture.index

#y = furniture['Sales'].resample('MS').mean()

from datetime import datetime
con=df_flights_new['Observed Timestamp']
df_flights_new['Observed Timestamp']=pd.to_datetime(df_flights_new['Observed Timestamp'])
df_flights_new.set_index('Observed Timestamp', inplace=True)

df_flights_new.index


y = df_flights_new['Fare'].resample('11h').mean()
#print(y)
#
#y.plot(figsize=(15, 6))
#plt.show()


y.plot(figsize=(15, 6))
plt.show()


#from pylab import rcParams
#rcParams['figure.figsize'] = 18, 8
#decomposition = sm.tsa.seasonal_decompose(y, model='additive')
#fig = decomposition.plot()
#plt.show()


from pylab import rcParams
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(y, model='additive',freq=3)
fig = decomposition.plot()
plt.show()




#p = d = q = range(0, 2)
#pdq = list(itertools.product(p, d, q))
#seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
#print('Examples of parameter combinations for Seasonal ARIMA...')
#print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
#print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
#print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
#print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

#
#for param in pdq:
#    for param_seasonal in seasonal_pdq:
#        try:
#            mod = sm.tsa.statespace.SARIMAX(y,
#                                            order=param,
#                                            seasonal_order=param_seasonal,
#                                            enforce_stationarity=False,
#                                            enforce_invertibility=False)
#results = mod.fit()
#print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
#        except:
#            continue
#

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,order=param,seasonal_order=param_seasonal,enforce_stationarity=False,enforce_invertibility=False)
            results = mod.fit()
                
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue



mod = sm.tsa.statespace.SARIMAX(y,
                                order=(0, 1, 1),
                                seasonal_order=(0, 0, 0, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])

#
#results.plot_diagnostics(figsize=(16, 8))
#plt.show()

results.plot_diagnostics(figsize=(16, 8))
plt.show()



#pred = results.get_prediction(start=pd.to_datetime('2017-01-01'), dynamic=False)
#pred_ci = pred.conf_int()
#ax = y['2014':].plot(label='observed')
#pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
#ax.fill_between(pred_ci.index,
#                pred_ci.iloc[:, 0],
#                pred_ci.iloc[:, 1], color='k', alpha=.2)
#ax.set_xlabel('Date')
#ax.set_ylabel('Furniture Sales')
#plt.legend()
#plt.show()

#y1 = pd.date_range('18/07/2017',periods=2500,freq='H')
#y1[0]
#y1[0].strftime('%Y/%m/%d')
#
#
pred = results.get_prediction(start=pd.to_datetime('2017-07-01'), dynamic=False)
pred_ci = pred.conf_int()
ax = y['2017':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Observed Date')
ax.set_ylabel('Fare')
plt.legend()
plt.show()


#y_forecasted = pred.predicted_mean
#y_truth = y['2017-01-01':]
#mse = ((y_forecasted - y_truth) ** 2).mean()
#print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))


y_forecasted = pred.predicted_mean
y_truth = y['2017-01-01':]
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))


















%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

df_flights = pd.read_csv('sample.csv')
print(df_flights.head())
print('\n Data Types:')
print(df_flights.dtypes)

#
#from datetime import datetime
#con=df_flights['Observed Timestamp']
#df_flights['Observed Timestamp']=pd.to_datetime(df_flights['Observed Timestamp'])
#df_flights.set_index('Observed Timestamp', inplace=True)
#check datatype of index

#df_flights.index

import plotly.offline as offline
import plotly.graph_objs as go
from plotly import tools
offline.offline.init_notebook_mode(connected=True)


#plot mean flight cost over time for each destination

df_filter_cost= df_flights[df_flights["Fare"]!=0]
df_date_airport= df_filter_cost.groupby(['Observed Timestamp','Arr'], as_index=False)["Fare"].mean()

data=[]
for airport in df_date_airport["Arr"].unique():
    df_airport=df_date_airport[df_date_airport["Arr"]==airport]
    data.append(go.Scatter(x=df_airport["Observed Timestamp"], y=df_airport["Fare"],name=airport))

layout = go.Layout(xaxis={
    'type': 'date',
    'tickformat': '%H:%M'
    })

layout = dict(
    title = "Average Flight Price Over Time by Destination",
    xaxis = dict(title='Date'),
    yaxis=dict(title='Flight Price($)'))

fig= dict(data=data, layout=layout)
offline.iplot(fig)


'''