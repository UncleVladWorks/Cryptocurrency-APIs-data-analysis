import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import datetime as dt
import time
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

key = ''

btc_id = '1182'

def get_btc():
    url = "https://min-api.cryptocompare.com/data/histoday?fsym=BTC&tsym=USD&limit=90&api_key="
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
    url = "https://min-api.cryptocompare.com/data/histoday?fsym=" + name + "&tsym=USD&limit=90&api_key="
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


btc = get_btc()
eth = get_other_crypto('ETH')

list = pd.concat([btc, eth], ignore_index=False, axis=1)
list['date'] = pd.to_datetime(list['time'], unit='s')
list2 = list.drop('time', axis=1)
date = list2['date']
list2.index = pd.MultiIndex.from_product([date])
list3 = list2.drop('date', axis=1)
price1 = list3['Price_BTC']


#creating dataframe
data = list2.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(list2)),columns=['date', 'Price_BTC'])
for i in range(0,len(data)):
	new_data['date'][i] = data['date'][i]
	new_data['Price_BTC'][i] = data['Price_BTC'][i]

#setting index
new_data.index = new_data.date
new_data.drop('date', axis=1, inplace=True)

#creating train and test sets
dataset = new_data.values

train = dataset[0:88,:]
valid = dataset[88:,:]

#converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(60,len(train)):
	x_train.append(scaled_data[i-60:i,0])
	y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

#predicting values, using past 60 from the train data
inputs = new_data[len(new_data) - len(valid) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)

X_test = []
for i in range(60,inputs.shape[0]):
	X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)
closing_price1 = scaler.inverse_transform(closing_price)

rms=np.sqrt(np.mean(np.power((valid-closing_price1),2)))

print(rms)
print(closing_price1)

df1 = pd.DataFrame(data=closing_price1)
df_date = list2.tail(3)
df_date2 = df_date['date']
df2 = df1
df2['Forecast'] = df1.reset_index(drop=True)

last = df_date2.index[-1]
last2 = last[0]
next= last2 + dt.timedelta(days=1)


dates=[]
for i in range(4): #add 7 days
	next= last2 + dt.timedelta(days=i)
	dates.append(next)

dates2 = pd.DataFrame(data=dates)
dates4 = dates2.astype('str')
dates3 = dates4.iloc[1:].reset_index(drop=True)
df3 = df2.drop(0, axis=1)
df4 = pd.concat([dates3, df3], axis=1)

fin = df4.to_json('/root/environments/Data/Forecasts/Price_BTC/Output_model2.json')

dates5 = dates2.iloc[1:].reset_index(drop=True)
df3.index = pd.MultiIndex.from_product([dates5[0]])

#for plotting
prev1 = list3.tail(10)
prev2 = prev1['Price_BTC']

list1 = pd.concat([prev2, df3], axis=1)

style.use('ggplot')

plt.figure(1)
list1['Price_BTC'].plot()
list1['Forecast'].plot()
plt.legend(loc='best') #bottom right
plt.xlabel('Date')
plt.ylabel('Price (US$)')
fig = plt.gcf()
fig.set_size_inches(14, 10)
fig.savefig('/root/environments/project01/static/data/fig_model2.png')
plt.clf()
