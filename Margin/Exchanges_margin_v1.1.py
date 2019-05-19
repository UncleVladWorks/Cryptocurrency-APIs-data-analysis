import requests
import pandas as pd
import seaborn as sns
import numpy as np
import os
import yagmail
import glob
from bittrex.bittrex import Bittrex
import cbpro
from poloniex_api import Poloniex
import krakenex
from pykrakenapi import KrakenAPI
import matplotlib
import matplotlib.pyplot as plt
from binance.client import Client
from os.path import join as pjoin
from API_keys import Keys
import time
from pycoingecko import CoinGeckoAPI
cg = CoinGeckoAPI()



pair = 'BTCUSD'
symbols = ['ETHBTC']

print("Getting initial price data")
try:
	#ETH and BTC price from GDAX
	def btc():
		client = cbpro.PublicClient()
		x = client.get_product_ticker(product_id='ETH-USD')
		for i in x:
			a = x['price']
			b = float(a)
		return b

	def btc_true():
		client = cbpro.PublicClient()
		x = client.get_product_ticker(product_id='BTC-USD')
		for i in x:
			a = x['price']
			b = float(a)
		return b
	x0 = btc_true()
except:
	print('Failed to load initial data')

print("Collecting prices from exchanges")

try:
	def binance(list):
		binance_api = Keys['binance_key']
		binance_secret = Keys['binance_secret']
		client = Client(binance_api, binance_secret)
		prices = client.get_all_tickers()
		for price in prices:
			if price['symbol'] in list:
				y = price['price']
				x = float(y)
				z = 1 / x
		a = btc()
		b = a * z
		return b
	x = binance(symbols)
except:
	x = 'Nan'

try:
	def polo():
		my_polo = Poloniex(
			API_KEY = Keys['polo1'],
			API_SECRET = Keys['polo2']
		)
		ticker = my_polo.returnTicker()
		for i in ticker:
			x = ticker['BTC_ETH']
			for v in x:
				y = x['last']
				z = float(y)
		a = btc()
		b = a / z
		return b
	y = polo()
except:
	y = 'Nan'

try:
	def bitlish():
		x = 'https://bitlish.com/api/v1/tickers'
		raw_data = requests.get(x).json()
		for i in raw_data:
			x1 = raw_data['btcusd']['last']
		return x1
	y2 = bitlish()
except:
	y2 = 'Nan'

try:
	def kraken():
		api = krakenex.API()
		k = KrakenAPI(api)
		x = k.get_ticker_information('XBTUSD')
		for i in x:
			a = x.iloc[0][0][0]
		return a
	y3 = kraken()
except:
	y3 = 'Nan'

try:
	def coinbase():
		public_client = cbpro.PublicClient()
		x = public_client.get_product_ticker(product_id='BTC-USD')
		for i in x:
			x1 = x['price']
		return x1
	y4 = coinbase()
except:
	y4 = 'Nan'

try:
	def okex():
		x = 'https://www.okex.com/api/spot/v3/instruments/ETH-BTC/ticker'
		raw_data = requests.get(x).json()
		for i in raw_data:
			x1 = raw_data['last']
			x2 = float(x1)
			z = 1 / x2
		a = btc()
		b = a * z
		return b
	y7 = okex()
except:
	y7 = 'Nan'

try:
	def hitbtc():
		x = 'https://api.hitbtc.com/api/2/public/ticker/BTCUSD'
		raw_data = requests.get(x).json()
		for i in raw_data:
			x = raw_data['last']
		return x
	y8 = hitbtc()
except:
	y8 = 'Nan'

# def bitz():
try:
	def coinbene():
		x = 'https://api.coinbene.com/v1/market/ticker?symbol=ethbtc'
		raw_data = requests.get(x).json()
		for i in raw_data:
			x1 = raw_data['ticker'][0]['last']
			x2 = float(x1)
			z = 1 / x2
		a = btc()
		b = a * z
		return b
	y10 = coinbene()
except:
	y10 = 'Nan'

try:
	def idax():
		x = 'https://openapi.idax.pro/api/v2/ticker?pair=ETH_BTC'
		raw_data = requests.get(x).json()
		for i in raw_data:
			x1 = raw_data['ticker'][0]['last']
			x2 = float(x1)
			z = 1 / x2
		a = btc()
		b = a * z
		return b
	y11 = idax()
except:
	y11 = 'Nan'

try:
	def huobi():
		x = 'https://api.huobi.pro/market/detail/merged?symbol=ethbtc'
		raw_data = requests.get(x).json()
		for i in raw_data:
			x1 = raw_data['tick']['close']
			x2 = float(x1)
			z = 1 / x2
		a = btc()
		b = a * z
		return b
	y12 = huobi()
except:
	y12 = 'Nan'

try:
	def coindeal():
		x = 'https://apigateway.coindeal.com/api/v1/public/orderbook/ETHBTC'
		raw_data = requests.get(x).json()
		for i in raw_data:
			x1 = raw_data['ask'][0]['price']
			x2 = float(x1)
			z = 1 / x2
		a = btc()
		b = a * z
		return b
	y13 = coindeal()
except:
	y13 = 'Nan'

try:
	def rightbtc():
		x = 'https://www.rightbtc.com/api/public/ticker/ETHBTC'
		raw_data = requests.get(x).json()
		for i in raw_data:
			x1 = raw_data['result']['last']
			x2 = float(x1)
			x3 = x2/1e8
			z = 1/x3
		a = btc()
		b = a * z
		return b
	y14 = rightbtc()
except:
	y14 = 'Nan'

try:
	def digifinex():
		x1 = cg.get_exchanges_tickers_by_id('digifinex')
		for i in x1:
			x2 = x1['tickers'][0]['converted_last']['usd']
		return x2
	y15 = digifinex()
except:
	y15 = 'Nan'

#Exchanges data variables
l = []
m = np.array(l, dtype = np.float32)
m = np.append(m, [x, y, y2, y4, y7, y8, y10, y11, y12, y13, y14, y15, y3])

df = pd.DataFrame(m)

def spread_ex(index, name):
	m1 = df.astype(float).round(2)
	for i in m1:
		a = m1.iloc[[index]]
	b = a.values
	b1 = b[0][0]
	b2 = float(b1)
	r1 = (m1.divide(b2))*100 - 100
	r2 = pd.DataFrame(data=r1)
	r3 = r2.rename(columns={0: name})
	return r3

# Constructing spread
exchanges = ['Binance', 'Poloniex', 'Bitlish', 'Coinbase', 'OKEX', 'HitBTC', 'Coinbene', 'IDAX', 'Huobi', 'Coindeal', 'RightBTC', 'Digifinex', 'Kraken']

a1 = spread_ex(0, 'Binance')
a2 = spread_ex(1, 'Poloniex')
# a3 = spread_ex(2, 'Bittrex')
a4 = spread_ex(2, 'Bitlish')
a5 = spread_ex(3, 'Coinbase')
a6 = spread_ex(4, 'OKEX')
a7 = spread_ex(5, 'HitBTC')
a8 = spread_ex(6, 'Coinbene')
a9 = spread_ex(7, 'IDAX')
a10 = spread_ex(8, 'Huobi')
a11 = spread_ex(9, 'Coindeal')
a12 = spread_ex(10, 'RightBTC')
a13 = spread_ex(11, 'Digifinex')
a14 = spread_ex(12, 'Kraken')

fin = pd.concat([a1, a2, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14], ignore_index=False, axis=1)
fin.index = pd.MultiIndex.from_product([exchanges])

# Creating picture
with sns.axes_style():
	ax = sns.heatmap(fin, square=False, annot=True, linewidths=1, cmap='RdYlGn', linecolor='white')
	ax.set_ylabel('SELL')
	ax.set_xlabel('BUY')
	ax.xaxis.set_label_position('top')
	fig = plt.gcf()
	fig.set_size_inches(16, 12)

print('Saving data')

# Assigning time
from datetime import datetime
t = datetime.now().strftime('%Y-%m-%d/')
t1 = datetime.now().strftime('%Y-%m-%d')

path = "/root/environments/Data/Exchanges_spreads/BTC"
path_pic = "/root/environments/project01/static/data/"

# Saving data
name_fin1 = 'Exchanges_Spreads_'
name_fin = name_fin1 + t1 + '.json'
path_to_file = pjoin(path, name_fin)
fin_csv = fin.to_json(path_or_buf=path_to_file, index=True)


png1 = 'last_img_exchanges_spreads.png'
p3 = path_pic
p3 += png1
fig.savefig(p3)
plt.clf()

print('Done')


