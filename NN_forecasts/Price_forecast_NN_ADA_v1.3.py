import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn import preprocessing, model_selection, metrics
from sklearn.ensemble import AdaBoostRegressor
from matplotlib import style
import datetime as dt
import time


key = ''

btc_id = '1182'

def get_btc():
    url = "https://min-api.cryptocompare.com/data/histoday?fsym=BTC&tsym=USD&limit=145&api_key="
    url += key
    f = requests.get(url).json()
    for exchange in f:
        ticker1 = f.get('Data')
        df = pd.DataFrame(data=ticker1)
    df1 = df.drop(['volumeto', 'high', 'low', 'open'], axis=1)
    n1 = 'Price_BTC'
    n2 = 'Volume_BTC'
    df2 = df1.rename(columns={'close': n1, 'volumefrom': n2})
    return df2

def get_other_crypto(name):
    url = "https://min-api.cryptocompare.com/data/histoday?fsym=" + name + "&tsym=USD&limit=145&api_key="
    url += key
    f = requests.get(url).json()
    for exchange in f:
        ticker1 = f.get('Data')
        df = pd.DataFrame(data=ticker1)
        df1 = df.drop(['time', 'volumeto', 'high', 'low', 'open'], axis=1)
    n1 = 'Price_' + name
    n2 = 'Volume_' + name
    df2 = df1.rename(columns={'close': n1, 'volumefrom': n2})
    return df2

def get_media():
    url = "https://min-api.cryptocompare.com/data/social/coin/histo/day?limit=145&api_key="
    url += key
    f = requests.get(url).json()
    for exchange in f:
        ticker1 = f.get('Data')
        df = pd.DataFrame(data=ticker1)
    df1 = df.drop(['time', 'code_repo_closed_issues', 'code_repo_forks', 'code_repo_open_pull_issues',
                   'code_repo_subscribers', 'markets_page_views', 'trades_page_views', 'code_repo_stars',
                   'reddit_posts_per_hour'], axis=1)
    return df1

def get_index( id, name):
    url2 = 'https://finance.yahoo.com/quote/%5E' + id + '/history/'
    DJIA = pd.read_html(url2, header=0)
    DJIA[0]=DJIA[0][:-1]
    dow1 = DJIA[0]['Close*']
    date1 = DJIA[0]['Date']
    date2 = pd.to_datetime(date1)
    date3 = pd.to_datetime(date2, format='%d%b%Y:%H:%M:%S.%f')
    dow1.index = pd.DatetimeIndex(date3)
    dow2 = dow1.astype(float)
    dow4 = dow2.resample('D', how='mean').fillna(method='ffill')
    dow6 = dow4.reset_index(drop=True)
    dow7 = dow6.to_frame()
    dow8 = dow7.rename(columns={'Close*': name})
    dow9 = dow8[:146]
    return dow9


btc = get_btc()
eth = get_other_crypto('ETH')
xrp = get_other_crypto('XRP')
dash = get_other_crypto('DASH')
tron = get_other_crypto('TRON')
vix = get_index('VIX', 'CBOE Volatility Index')
btc_media = get_media()

list = pd.concat([btc, eth, xrp, vix, dash, tron, btc_media], ignore_index=False, axis=1)
list['date'] = pd.to_datetime(list['time'], unit='s')
list2 = list.drop('time', axis=1)
date = list2['date']
list2.index = pd.MultiIndex.from_product([date])
list3 = list2.drop('date', axis=1)

forecast_col = 'Price_BTC'

n = 5  # days
# n=0.01*len(df) #shift is 1% of length of df
forecast_out = int(math.ceil(n))  # make n integer
price2 = list3
price2['label'] = list3[forecast_col].shift(-forecast_out)

df = price2
forecast_col = 'Price_BTC'
df.fillna(-99999, inplace=True)
# n=0.01*len(df)

df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]  # most recent data (e.g. 95-100)
X = X[:-forecast_out]  # data until beginning of X_lately (e.g. 0-94)

df.dropna(inplace=True)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = AdaBoostRegressor(n_estimators=500, learning_rate=0.3)
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print('accuracy is %f' % accuracy)
forecast_set = clf.predict(X_lately)
print(forecast_set, accuracy, forecast_out)

list3 = list.drop('time', axis=1)
list3.index = pd.MultiIndex.from_product([date])
list4 = list3.drop('date', axis=1)
list5 = list4.tail(7)

list5['Forecast'] = np.nan  # create a new column for prediction and fill up with nan

last_date = list5.iloc[-1].name
last_date2 = last_date[0]
last_unix = last_date2.timestamp()
one_day = 86400  # seconds
next_unix = last_unix + one_day  # next day

###PARSE x-axis into date format
for i in forecast_set:
	next_date = dt.datetime.fromtimestamp(next_unix)
	next_unix += one_day
	list5.loc[next_date] = [np.nan for _ in range(len(list5.columns) - 1)] + [i]  # df.loc is the index location

# Data shift
lis1 = list5.shift(-1)
lis2 = lis1.drop(lis1.tail(1).index)

#  Creating pathes
path_fig = '/root/environments/project01/static/data/'
# path_fig = ''
path_file = '/root/environments/Data/Forecasts/Price_BTC/'
# path_file = ''

# Plotting main list with Real price and Predictions in future
style.use('ggplot')
plt.figure(1)
list5['Price_BTC'].plot()
lis2['Forecast'].plot()
plt.legend(loc='best')  # bottom right
plt.xlabel('Date')
plt.ylabel('Price (US$)')
fig = plt.gcf()
fig.set_size_inches(14, 10)
path1 = path_fig
fig.savefig(path1 + 'fig_model3.png')
plt.clf()

# Creating dataframe of Predictions
list_add = list5.tail(7).shift(-1)
list_add1 = list_add.drop(list_add.tail(1).index)
df_ex = list_add1.tail(5)
df_ex1 = df_ex['Forecast'].reset_index(drop=True)
df_ex2 = list_add1.tail(7)

last = df_ex2.index[0]
last2 = last[0]
next = last2 + dt.timedelta(days=1)

dates = []
for i in range(6):  # add 7 days
	next = last2 + dt.timedelta(days=i)
	dates.append(next)

dates2 = pd.DataFrame(data=dates)
dates4 = dates2.astype('str')
dates3 = dates4.iloc[1:].reset_index(drop=True)

# Saving data
df_ex3 = pd.concat([dates3, df_ex1], axis=1, ignore_index=True).rename(columns={0: 'Date', 1: 'Prediction'})
fin = df_ex3.to_json(path_file + 'Output_model3.json')

# Reading csv and saving new data there
rdf = pd.read_csv(path_file + 'Output_model3.csv')

y_pred1 = rdf

# Condition to properly save the data
df_com1 = df_ex3.tail(1)
df_com2 = df_com1['Date'].reset_index(drop=True)
df_com3 = df_com2[0]
rdf_com1 = rdf.tail(1)
rdf_com2 = rdf_com1['Date'].reset_index(drop=True)
rdf_com3 = rdf_com2[0]
if df_com3 == rdf_com3:
	df_ex4 = df_ex3.tail(1)
else:
	df_ex4 = df_ex3.tail(2)

# Saving data into csv file
rdf1 = pd.merge(rdf, df_ex4, on=['Date', 'Prediction'], how='right')
rdf.drop(rdf.tail(1).index, inplace=True)
rdf2 = pd.merge(rdf, rdf1, on=['Date', 'Prediction'], how='outer')
rdf4 = rdf2.to_csv(path_file + 'Output_model3.csv', index=False)

# Creating list of compare
rdf_date = rdf2['Date']
date_time = pd.to_datetime(rdf_date)
rdf2.index = pd.MultiIndex.from_product([date_time])
list6 = list5['Price_BTC']
list7 = list4.tail(31)
compare = pd.concat([list7, rdf2], axis=1)

# Datasets for Rms calculation
y_true1 = list7['Price_BTC']
y_true2 = y_true1.tail(3)
y_pred2 = y_pred1.drop(y_pred1.tail(1).index)
y_pred3 = y_pred2.tail(3)
y_pred3.index = pd.MultiIndex.from_product([y_pred3['Date']])
y_pred4 = y_pred3['Prediction'].round(2)

rms = metrics.mean_squared_error(y_true2, y_pred4)
print(rms)

# Plotting comparison between Predictions and Real prices
plt.figure(2)
compare['Price_BTC'].plot()
compare['Prediction'].plot()
plt.legend(loc='best')  # bottom right
plt.xlabel('Date')
plt.ylabel('Price (US$)')
fig = plt.gcf()
fig.set_size_inches(14, 8)
fig.savefig(path_fig + 'fig_model1_compare_03.png')
plt.clf()

# Create output specification
text_file = open(path_file + "Model_specification_03.txt", "w")
text_file.write('Current accuracy: %f' % accuracy)
text_file.write(' percent. Current Rms: %f' % rms)
text_file.close()
