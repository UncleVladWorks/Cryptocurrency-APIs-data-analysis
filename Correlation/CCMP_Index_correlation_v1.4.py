import requests
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import time

key = ''

btc_id = '1182'

def get_other_crypto(name):
    url = "https://min-api.cryptocompare.com/data/histoday?fsym=" + name + "&tsym=USD&limit=147&api_key="
    url += key
    f = requests.get(url).json()
    for exchange in f:
        ticker1 = f.get('Data')
        df = pd.DataFrame(data=ticker1)
    df1 = df.drop(['time', 'volumeto', 'high', 'low', 'open', 'volumefrom'], axis=1)
    df2 = df1.rename(columns={'close': name})
    return df2

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
    dow5 = dow4.rename(columns={'Close*': name})
    dow6 = dow4.reset_index(drop=True)
    dow7 = dow6.to_frame()
    dow8 = dow7.rename(columns={'Close*': name})
    return dow8, dow5

def get_index_n225( id, name):
    url2 = 'https://finance.yahoo.com/quote/%5E' + id + '/history/'
    DJIA = pd.read_html(url2, header=0)
    DJIA[0]=DJIA[0][:-1]
    dow1 = DJIA[0]['Close*']
    date1 = DJIA[0]['Date']
    date2 = pd.to_datetime(date1)
    date3 = pd.to_datetime(date2, format='%d%b%Y:%H:%M:%S.%f')
    dow1.index = pd.DatetimeIndex(date3)
    dow1.replace('-', None, inplace=True)
    dow2 = dow1.astype(float)
    dow4 = dow2.resample('D', how='mean').fillna(method='ffill')
    dow5 = dow4.rename(columns={'Close*': name})
    dow6 = dow4.reset_index(drop=True)
    dow7 = dow6.to_frame()
    dow8 = dow7.rename(columns={'Close*': name})
    return dow8, dow5

# Crypto
print('Collecting data for currencies')
btc = get_other_crypto('BTC')
eth = get_other_crypto('ETH')
xrp = get_other_crypto('XRP')
eos = get_other_crypto('EOS')
ltc = get_other_crypto('LTC')
tron = get_other_crypto('TRX')
bch = get_other_crypto('BCH')
zec = get_other_crypto('ZEC')
neo = get_other_crypto('NEO')
okex = get_other_crypto('OKB')
ada = get_other_crypto('ADA')
etc = get_other_crypto('ETC')

# Indexes
print('Collecting data for indexes')
sp500, sp500_d = get_index('GSPC', 'S&P 500')
dow, dow_d = get_index('DJI', 'Dow Jones')
nsdq, nsdq_d = get_index('IXIC', 'NASDAQ')
rus2000, rus2000_d = get_index('RUT', 'Russel 2000')
vix, vix_d = get_index('VIX', 'CBOE Volatility Index')
n225, n225_d = get_index_n225('N225', 'Nikkei 225')

# Combining everything
list_crypto = pd.concat([btc, eth, xrp, eos, ltc, tron, bch, zec, neo, okex, ada, etc], ignore_index=False, axis=1)
list_index = pd.concat([sp500, dow, nsdq, rus2000, vix, n225], ignore_index=False, axis=1)
list_index2 = list_index.shift(1)
list_index3 = list_index.shift(3)

list_all1 = pd.concat([list_index, list_crypto], ignore_index=False, axis=1)
list_all1 = list_all1[:30]
list_all2 = pd.concat([list_index2, list_crypto], ignore_index=False, axis=1)
list_all2 = list_all2[:30]
list_all3 = pd.concat([list_index3, list_crypto], ignore_index=False, axis=1)
list_all3 = list_all3[:30]

# Calculating correlation
corr1 = list_all1.pct_change().corr(method='pearson')
corr11 = corr1[6:].round(2).iloc[:, :6]

corr2 = list_all2.pct_change().corr(method='pearson')
corr21 = corr2[6:].round(2).iloc[:, :6]

corr3 = list_all3.pct_change().corr(method='pearson')
corr31 = corr3[6:].round(2).iloc[:, :6]

# Saving csvs and pics
t1 = datetime.datetime.now().strftime('%Y-%m-%d')

# Index graphs
names = ['Nikkei 225', 'S&P 500', 'Dow Jones', 'NASDAQ', 'Russel 2000', 'CBOE Volatility Index']

# Adjusting values for graphs
vix_d1 = vix_d * 1000
nsdq_d1 = nsdq_d * 3
sp500_d1 = sp500_d * 10
rus2000_d1 = rus2000_d * 20

lip1 = pd.concat([n225_d, sp500_d1, dow_d, nsdq_d1, rus2000_d1, vix_d1], ignore_index=False, axis=1)
lip2 = lip1.rename(columns={0:names[0], 1:names[1], 2:names[2], 3:names[3], 4:names[4], 5:names[5]})
lip3 = lip2.tail(31)

path = '/root/environments/Data/GeneralAnalytics/IndexCor/Month/'
path_pic = '/root/environments/project01/static/data/'

try:
	os.makedirs(path)
except FileExistsError:
	"Directory already exists"
	pass

print('Saving images')

with sns.axes_style("white"):
	ax = sns.heatmap(corr11, vmin=-1, vmax=1, square=True,  cmap="YlGnBu", annot=True)
	ax.set_title('Day to Day Index correlation / Month')
	fig = ax.get_figure()
	fig.set_size_inches(8, 8)
png1 = 'Day_to_Day_correlation_Month'
png = png1 + '.png'
p3 = path_pic
p3 += png
fig.savefig(p3)
plt.clf()

with sns.axes_style("white"):
	bx = sns.heatmap(corr21, vmin=-1, vmax=1, square=True, cmap="YlGnBu", annot=True)
	bx.set_title('Day to Yesterday Index correlation / Month')
	fig1 = bx.get_figure()
	fig1.set_size_inches(8, 8)
png2 = 'Day_to_Yesterday_correlation_Month'
png3 = png2 + '.png'
p4 = path_pic
p4 += png3
fig1.savefig(p4)
plt.clf()

with sns.axes_style("white"):
	cx = sns.heatmap(corr31, vmin=-1, vmax=1, square=True, cmap="YlGnBu", annot=True)
	cx.set_title('Day to 3 Days Index correlation / Month')
	fig2 = cx.get_figure()
	fig2.set_size_inches(8, 8)
png4 = 'Day_to_3Days_correlation_Month'
png5 = png4 + '.png'
p5 = path_pic
p5 += png5
fig2.savefig(p5)
plt.clf()

plt.figure(1)
lip2.plot()
plt.legend(loc='best')  # bottom right
plt.xlabel('Time')
plt.ylabel('Index Value Adj.')
fig3 = plt.gcf()
fig3.set_size_inches(12, 10)
png21 = 'Index values graphs.png'
p6 = path_pic
p6 += png21
fig3.savefig(p6)
plt.clf()

plt.figure(2)
lip3.plot()
plt.legend(loc='best')  # bottom right
plt.xlabel('Time')
plt.ylabel('Index Value Adj.')
fig4 = plt.gcf()
fig4.set_size_inches(12, 10)
png22 = 'Index values graphs 02.png'
p7 = path_pic
p7 += png22
fig4.savefig(p7)
plt.clf()

print('Saving csvs')

p1 = path
name_m1 = 'Day_to_Day_correlation_Month_'
name_m = name_m1 + t1 + '.csv'
p1 += name_m
csv = corr11.to_csv(path_or_buf=p1, index=True)
p2 = path
name_m2 = 'Day_to_Yesterday_correlation_Month_'
name_m3 = name_m2 + t1 + '.csv'
p2 += name_m3
csv1 = corr21.to_csv(path_or_buf=p2, index=True)
p21 = path
name_m4 = 'Day_to_3Days_correlation_Month_'
name_m5 = name_m4 + t1 + '.csv'
p21 += name_m5
csv2 = corr31.to_csv(path_or_buf=p21, index=True)

print('Done')
