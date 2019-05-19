import requests
import pandas as pd
import time
import numpy as np
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from matplotlib import style
import datetime

api_key = ''

# Recieving names of all crypto on coincompare
def get_names():
    url = "https://min-api.cryptocompare.com/data/all/coinlist"
    f = requests.get(url).json()
    for exchange in f:
        ticker1 = f.get('Data')
    for key, value in ticker1.items():
        if type(value) is dict:
            yield (key, value['FullName'])
        else:
            yield (value)

list_n = []
list_s = []

# Converting and concentrating them into readable format
def convert(s):
    new = ""
    c = ','
    for x in s:
        new += x
        new += c
    return new

# Getting the data for all cryptocurrencies
def get_data(id):
    url1 = 'https://min-api.cryptocompare.com/data/pricemultifull?fsyms=' + id + '&tsyms=USD&api_key='
    url1 += api_key
    f1 = requests.get(url1).json()
    for exchange in f1:
        ticker2 = f1.get('RAW')

        def gen(data):
            for k, t in ticker2.items():
                if type(t) is dict:
                    yield (k, t['USD'][data])

    list_pr = []
    list_nm = []
    list_vol = []
    list_cap = []
    list_high = []
    list_low = []
    list_pr_last = []

    for k, t in gen('CHANGEPCTDAY'):
        list_pr.append(t)
    for k, t in gen('FROMSYMBOL'):
        list_nm.append(t)
    for k, t in gen('TOTALVOLUME24HTO'):
        list_vol.append(t)
    for k, t in gen('MKTCAP'):
        list_cap.append(t)
    for k, t in gen('HIGH24HOUR'):
        list_high.append(t)
    for k, t in gen('LOW24HOUR'):
        list_low.append(t)
    for k, t in gen('PRICE'):
        list_pr_last.append(t)

    return list_pr, list_nm, list_vol, list_cap, list_high, list_low, list_pr_last

def get_all(lists):
    list_pr1 = []
    list_nm1 = []
    list_vol1 = []
    list_cap1 = []
    list_high1 = []
    list_low1 = []
    list_pr_last1 = []
    length = len(lists)
    i = 0
    while i < length:
        x1, x2, x3, x4, x5, x6, x7 = get_data(lists[i])
        list_pr1.append(x1)
        list_nm1.append(x2)
        list_vol1.append(x3)
        list_cap1.append(x4)
        list_high1.append(x5)
        list_low1.append(x6)
        list_pr_last1.append(x7)
        i += 1
    else:
        return list_pr1, list_nm1, list_vol1, list_cap1, list_high1, list_low1, list_pr_last1

# Flat out the outcome
def flat(lists):
    list_new = [item for sublist in lists for item in sublist]
    return list_new

def get_crypto(name):
    try:
        url = "https://min-api.cryptocompare.com/data/histohour?fsym=" + name + "&tsym=USD&limit=12&api_key="
        url += api_key
        f = requests.get(url).json()
        for exchange in f:
            ticker1 = f.get('Data')
            df = pd.DataFrame(data=ticker1)
        df1 = df.drop(['volumeto', 'volumefrom', 'high', 'low', 'open'], axis=1)
        df1['date'] = pd.to_datetime(df1['time'], unit='s')
        df2 = df1.drop(['time', 'date'], axis=1)
        date = df1['date']
        df2.index = pd.DatetimeIndex(date)
        n1 = name
        df3 = df2.rename(columns={'close': n1})
        return df3, n1
    except:
        pass

def get_crypto_volume(name):
    try:
        url = "https://min-api.cryptocompare.com/data/histohour?fsym=" + name + "&tsym=USD&limit=12&api_key="
        url += api_key
        f = requests.get(url).json()
        for exchange in f:
            ticker1 = f.get('Data')
            df = pd.DataFrame(data=ticker1)
        df1 = df.drop(['close', 'volumefrom', 'high', 'low', 'open'], axis=1)
        df1['date'] = pd.to_datetime(df1['time'], unit='s')
        df2 = df1.drop(['time', 'date'], axis=1)
        date = df1['date']
        df2.index = pd.DatetimeIndex(date)
        n2 = name
        df_vol = df2.rename(columns={'volumeto': n2})
        return df_vol, n2
    except:
        pass


print('Collecting data')
for key, value in get_names():
	list_n.append(value)
	list_s.append(key)

list_s7 = convert(list_s)
list_s8 = list_s7[0:1000]
list_s9 = list_s7[1001:2001]
list_s10 = list_s7[2002:3001]
list_s11 = list_s7[3002:3999]
list_s12 = list_s7[4000:4999]
list_s13 = list_s7[4999:5999]
list_s14 = list_s7[6005:7002]
list_s15 = list_s7[7002:8002]
list_s16 = list_s7[8003:9002]
list_s17 = list_s7[9003:10001]
list_s18 = list_s7[10002:10998]
list_s19 = list_s7[10999:11999]
list_s20 = list_s7[12000:12998]
list_s21 = list_s7[13002:14001]
list_s22 = list_s7[14002:15000]
list_s23 = list_s7[15001:16001]
list_s24 = list_s7[16002:17002]
list_s25 = list_s7[17004:18000]

list_all = [list_s8, list_s9, list_s10, list_s11, list_s12, list_s13, list_s14, list_s15, list_s16, list_s17, list_s18,
			list_s19, list_s20, list_s21, list_s22, list_s23, list_s24, list_s25]

list_pr1, list_nm1, list_vol1, list_cap1, list_high1, list_low1, list_pr_last2 = get_all(list_all)

print('Data received')

list_nm2 = flat(list_nm1)
list_pr2 = flat(list_pr1)
list_vol2 = flat(list_vol1)
list_cap2 = flat(list_cap1)
list_high2 = flat(list_high1)
list_low2 = flat(list_low1)
list_pr_last3 = flat(list_pr_last2)

pd.options.display.float_format = '{:.2f}'.format

# Calculating overall volatility
xh1 = pd.DataFrame(data=list_high2)
xl1 = pd.DataFrame(data=list_low2)
xhl1 = pd.concat([xh1, xl1], axis=1, ignore_index=True)
xhl2 = (xhl1[0] / xhl1[1] - 1)
vlot1 = xhl2.std(axis=0)
vlot11 = vlot1 ** .5
vlot12 = (vlot11 * vlot1).round(2)
vlot2 = str(vlot12)

volat1 = xh1.sum(axis=0)
volat2 = xl1.sum(axis=0)
volat3 = ((volat2 - volat1) / volat1) * 100
volat4 = volat3[0].round(2)
volat5 = str(volat4)

# Concentrating lists into dataframes
t1 = datetime.datetime.now().strftime('%Y-%m-%d')
t2 = str(t1)

x1 = pd.DataFrame(data=list_pr2)
x2 = pd.DataFrame(data=list_vol2)
x3 = pd.DataFrame(data=list_cap2)
xpr1 = pd.DataFrame(data=list_pr_last3)
x4 = pd.concat([x1, x2, x3, xhl2, xpr1], axis=1, ignore_index=True)
x4.index = pd.MultiIndex.from_product([list_nm2])
x5 = x4.loc[(x4[1] > 250000)]

# Saving and extracting data
path = '/root/environments/Data/News/Report_day/'

xnm = pd.DataFrame(data=list_nm2).rename(columns={0: 'Coins'})

# Save PRICE data
xpr1 = pd.concat([xnm, xpr1], axis=1, ignore_index=False)
xpr2 = xpr1.rename(columns={0: t2})
# xpr_csv = xpr2.to_csv(path+'Price_output.csv', index=False)
xpr_read = pd.read_csv(path + 'Price_output.csv')
xpr_read_name = xpr_read.columns.values
xpr_read_name1 = xpr_read_name[-1]
if t2 != xpr_read_name1:
	xpr3 = pd.merge(xpr_read, xpr2, on=['Coins'], how='outer')
else:
	xpr3 = xpr_read
xpr31 = xpr3.drop_duplicates(subset='Coins', keep=False)
xpr_write = xpr31.to_csv(path + 'Price_output.csv', index=False)

# Save VOLUME data
xvol1 = pd.concat([xnm, x2], axis=1, ignore_index=False)
xvol2 = xvol1.rename(columns={0: t2})
# xvol_csv = xvol2.to_csv(path+'Volume_output.csv', index=False)
xvol_read = pd.read_csv(path + 'Volume_output.csv')
xvol_read_name = xvol_read.columns.values
xvol_read_name1 = xvol_read_name[-1]
if t2 != xvol_read_name1:
	xvol3 = pd.merge(xvol_read, xvol2, on=['Coins'], how='outer')
else:
	xvol3 = xvol_read
xvol31 = xvol3.drop_duplicates(subset='Coins', keep=False)
xvol_write = xvol31.to_csv(path + 'Volume_output.csv', index=False)

# Save MARKET CAP data
xcap1 = pd.concat([xnm, x3], axis=1, ignore_index=False)
xcap2 = xcap1.rename(columns={0: t2})
# xcap_csv = xcap2.to_csv(path+'MCap_output.csv', index=False)
xcap_read = pd.read_csv(path + 'MCap_output.csv')
xcap_read_name = xcap_read.columns.values
xcap_read_name1 = xcap_read_name[-1]
if t2 != xcap_read_name1:
	xcap3 = pd.merge(xcap_read, xcap2, on=['Coins'], how='outer')
else:
	xcap3 = xcap_read
xcap31 = xcap3.drop_duplicates(subset='Coins', keep=False)
xcap_write = xcap31.to_csv(path + 'MCap_output.csv', index=False)

# Calculate PRICE data
xpr_read1 = pd.read_csv(path + 'Price_output.csv')
y = xpr_read1[xpr_read1.columns[-2:]]
y_name = y.columns.values
y_name1 = y_name[0]
y_name2 = y_name[1]
y['PR_CHG'] = ((y[y_name2] - y[y_name1]) / y[y_name1]) * 100
y1 = y['PR_CHG']

# Calculate VOLUME data
xvol_read1 = pd.read_csv(path + 'Volume_output.csv')
v = xvol_read1[xvol_read1.columns[-2:]]
v_name = v.columns.values
v_name1 = v_name[0]
v_name2 = v_name[1]
v['VOL_CHG'] = ((v[v_name2] - v[v_name1]) / v[v_name1]) * 100
v1 = v['VOL_CHG']

# Calculate MCAP data
xcap_read1 = pd.read_csv(path + 'MCap_output.csv')
z = xcap_read1[xcap_read1.columns[-2:]]
z_name = z.columns.values
z_name1 = z_name[0]
z_name2 = z_name[1]
z['CAP_CHG'] = ((z[z_name2] - z[z_name1]) / z[z_name1]) * 100
z1 = z['CAP_CHG']

x_app = pd.concat([y1, x2, x3, xhl2, xpr1, v1, z1], axis=1, ignore_index=True)
x_app.index = pd.MultiIndex.from_product([x_app[4]])
x_app1 = x_app.rename(columns={4: 'coins'})
x_app2 = x_app1.drop('coins', axis=1)
x_app3 = x_app2.drop(x_app2.tail(14).index)
x_app4 = x_app3.loc[(x_app3[1] > 250000)]

# Sorting values by different columns
x21 = x5.sort_values(by=0, axis=0, ascending=False)
x22 = x5.sort_values(by=1, axis=0, ascending=False)
x23 = x5.sort_values(by=2, axis=0, ascending=False)
x24 = x5.abs()
x241 = x24.sort_values(by=3, axis=0, ascending=False)

x2411 = x_app4.replace([np.inf, -np.inf], np.nan)
x242 = x2411.sort_values(by=6, axis=0, ascending=False)
x243 = x2411.sort_values(by=7, axis=0, ascending=False)

# Sum of prices
df2 = x22.head(50)
df3 = x21.head(50)
df_sum1 = df2[0]
df_sum2 = df_sum1.sum(axis=0)

# Prices
df31 = x21.tail(3).index.values
df32 = x21.head(3).index.values
df42 = ''.join(map(str, df31))
df43 = df42.replace('(', '').replace(')', '')
df52 = ''.join(map(str, df32))
df53 = df52.replace('(', '').replace(')', '')

df311 = x21.tail(3)
df312 = df311[0].values.round(2) * (-1)
df313 = ', '.join(map(str, df312))

df411 = x21.head(3)
df412 = df411[0].values.round(2)
df413 = ', '.join(map(str, df412))

# Volatility
df_vol1 = x241.head(3).index.values
df_vol2 = ''.join(map(str, df_vol1))
df_vol3 = df_vol2.replace('(', '').replace(')', '')
df_vol4 = x241.head(3)
df_vol5 = df_vol4[3].values.round(2)
df_vol6 = ', '.join(map(str, df_vol5))

# Volume change
v_sum1 = v[v_name1].sum(axis=0)
v_sum2 = v[v_name2].sum(axis=0)
v_sum3 = (((v_sum2 - v_sum1) / v_sum1) * 100).round(2)
v_sum4 = np.absolute(v_sum3)
v_sum5 = str(v_sum4)

x2422 = x242[6]
x2423 = x2422.dropna(axis=0)

vol_c1 = x2423.tail(3).index.values
vol_c2 = ''.join(map(str, vol_c1))
vol_c3 = vol_c2.replace('(', '').replace(')', '')

vol_a1 = x2423.head(3).index.values
vol_a2 = ''.join(map(str, vol_a1))
vol_a3 = vol_a2.replace('(', '').replace(')', '')

# MCap change
cap_sum1 = z[z_name1].sum(axis=0)
cap_sum2 = z[z_name2].sum(axis=0)
cap_sum3 = ((cap_sum2 - cap_sum1) / cap_sum1) * 100
cap_sum31 = cap_sum3.round(2)
cap_sum4 = str(cap_sum31)

print('Data calculated')

# Text arguments
pos1 = 'положительную'
pos2 = 'наибольший рост'
pos3 = 'вырос'
pos31 = 'выросла'
pos4 = 'более'
neg1 = 'отрицательную'
neg2 = 'наибольшее падение'
neg3 = 'упал'
neg31 = 'упала'
neg4 = 'менее'

# Condition
ans11 = df_vol3
ans12 = df_vol6
if df_sum2 < 0:
	ans1 = neg1
	ans2 = neg2
	ans3 = df43
	ans4 = neg3
	ans5 = df313
	ans6 = pos4
	ans7 = pos2
	ans8 = df53
	ans9 = pos3
	ans10 = df413

	x1 = df31
	x2 = df32
else:
	ans1 = pos1
	ans2 = pos2
	ans3 = df53
	ans4 = pos3
	ans5 = df413
	ans6 = neg4
	ans7 = neg2
	ans8 = df43
	ans9 = neg3
	ans10 = df313

	x1 = df32
	x2 = df31

if v_sum3 < 0:
	ans13 = neg3
	ans14 = neg2
	ans15 = vol_c3

	vol51 = x2423.head(3).values.round(2)
	vol53 = ', '.join(map(str, vol51))
	x3 = vol_c1
else:
	ans13 = pos3
	ans14 = pos2
	ans15 = vol_a3

	vol51 = x2423.head(3).values.round(2)
	vol53 = ', '.join(map(str, vol51))
	x3 = vol_a1

# Open and write text file
text_file = open("/root/environments/Data/News/Report_day/text/Report_day_1_pr1.txt", "w")
text_file.write("Рынок цен криптовалют в долларах США имел " + ans1 + ' динамику за последний день. \n')
text_file.write('Из них ' + ans2 + ' испытали: ' + ans3 + ' каждый из которых ' + ans4 + ' на: ' + ans5 + ' процентных пункта соответственно.\n')
text_file.close()

text_file = open("/root/environments/Data/News/Report_day/text/Report_day_2_pr2.txt", "w")
text_file.write("Однако, другие криптовалюты были " + ans6 + ' успешны за эти сутки. \n')
text_file.write(
	'Из них ' + ans7 + ' испытали: ' + ans8 + ' каждый из которых ' + ans9 + ' на: ' + ans10 + ' процентных пункта соответственно.\n')
text_file.close()

text_file = open("/root/environments/Data/News/Report_day/text/Report_day_3_vol.txt", "w")
text_file.write('Общая волатильность рынка: ' +volat5+ '%. Суммарная волатильность криптовавлют составила: ' + vlot2 + '% \n')
text_file.write('Тремя самыми волатильными криптовалютами оказались: ' + ans11 + ' с процентными коэффициентами: ' + ans12 + ' соответсвенно  \n')
text_file.close()

text_file = open("/root/environments/Data/News/Report_day/text/Report_day_4.txt", "w")
text_file.write('Объем торгов на рынке криптовалют в долларах США ' + ans13 + ' на: ' + v_sum5 + '% \n')
text_file.write('Три монеты, испытавшие ' + ans14 + ': ' + ans15 + ' с процентными коэффициентами: ' + vol53 + ' соответсвенно \n')
text_file.close()

text_file = open("/root/environments/Data/News/Report_day/text/Report_day_5.txt", "w")
text_file.write('Общая капитализация рынка криптовалют в долларах США изменилась на: ' + cap_sum4 + '% \n \n')
text_file.write('Источник: coincompare.com. Данные взяты о криптовалютах с дневным объемом торгов больше $250000 \n')
text_file.close()

print('Text created')

style.use('ggplot')

plt1 = ''.join(map(str, x1[0]))
plt2 = ''.join(map(str, x1[1]))
plt3 = ''.join(map(str, x1[2]))
plt4 = ''.join(map(str, x2[0]))
plt5 = ''.join(map(str, x2[1]))
plt6 = ''.join(map(str, x2[2]))
plt7 = ''.join(map(str, x3[0]))
plt8 = ''.join(map(str, x3[1]))
plt9 = ''.join(map(str, x3[2]))

path1 = '/root/environments/project01/static/data/news/daily/'
try:
	y1, n1 = get_crypto(plt1)
	plt.figure(1)
	y1[n1].plot()
	plt.legend(loc='best')  # bottom right
	plt.xlabel('Time')
	plt.ylabel('Price (US$)')
	fig = plt.gcf()
	fig.set_size_inches(8, 6)
	png1 = 'Graph_01.png'
	p1 = path1
	p1 += png1
	fig.savefig(p1)
	plt.clf()
except:
	pass

try:
	y2, n2 = get_crypto(plt2)
	plt.figure(2)
	y2[n2].plot(color='blue')
	plt.legend(loc='best')  # bottom right
	plt.xlabel('Time')
	plt.ylabel('Price (US$)')
	fig = plt.gcf()
	fig.set_size_inches(8, 6)
	png2 = 'Graph_02.png'
	p2 = path1
	p2 += png2
	fig.savefig(p2)
	plt.clf()
except:
	pass
try:
	y3, n3 = get_crypto(plt3)
	plt.figure(3)
	y3[n3].plot(color='green')
	plt.legend(loc='best')  # bottom right
	plt.xlabel('Time')
	plt.ylabel('Price (US$)')
	fig = plt.gcf()
	fig.set_size_inches(8, 6)
	png3 = 'Graph_03.png'
	p3 = path1
	p3 += png3
	fig.savefig(p3)
	plt.clf()
except:
	pass
try:
	y4, n4 = get_crypto(plt4)
	plt.figure(4)
	y4[n4].plot(color='red')
	plt.legend(loc='best')  # bottom right
	plt.xlabel('Time')
	plt.ylabel('Price (US$)')
	fig = plt.gcf()
	fig.set_size_inches(8, 6)
	png4 = 'Graph_04.png'
	p4 = path1
	p4 += png4
	fig.savefig(p4)
	plt.clf()
except:
	pass
try:
	y5, n5 = get_crypto(plt5)
	plt.figure(5)
	y5[n5].plot(color='blue')
	plt.legend(loc='best')  # bottom right
	plt.xlabel('Time')
	plt.ylabel('Price (US$)')
	fig = plt.gcf()
	fig.set_size_inches(8, 6)
	png5 = 'Graph_05.png'
	p5 = path1
	p5 += png5
	fig.savefig(p5)
	plt.clf()
except:
	pass
try:
	y6, n6 = get_crypto(plt6)
	plt.figure(6)
	y6[n6].plot(color='green')
	plt.legend(loc='best')  # bottom right
	plt.xlabel('Time')
	plt.ylabel('Price (US$)')
	fig = plt.gcf()
	fig.set_size_inches(8, 6)
	png6 = 'Graph_06.png'
	p6 = path1
	p6 += png6
	fig.savefig(p6)
	plt.clf()
except:
	pass

try:
	y7, n7 = get_crypto_volume(plt7)
	plt.figure(7)
	y7[n7].plot(color='red')
	plt.legend(loc='best')  # bottom right
	plt.xlabel('Time')
	plt.ylabel('Volume (US$)')
	fig = plt.gcf()
	fig.set_size_inches(8, 6)
	png7 = 'Graph_07.png'
	p7 = path1
	p7 += png7
	fig.savefig(p7)
	plt.clf()
except:
	pass

try:
	y8, n8 = get_crypto_volume(plt8)
	plt.figure(8)
	y8[n8].plot(color='blue')
	plt.legend(loc='best')  # bottom right
	plt.xlabel('Time')
	plt.ylabel('Volume (US$)')
	fig = plt.gcf()
	fig.set_size_inches(8, 6)
	png8 = 'Graph_08.png'
	p8 = path1
	p8 += png8
	fig.savefig(p8)
	plt.clf()
except:
	pass

try:
	y9, n9 = get_crypto_volume(plt9)
	plt.figure(9)
	y9[n9].plot(color='green')
	plt.legend(loc='best')  # bottom right
	plt.xlabel('Time')
	plt.ylabel('Volume (US$)')
	fig = plt.gcf()
	fig.set_size_inches(8, 6)
	png9 = 'Graph_09.png'
	p9 = path1
	p9 += png9
	fig.savefig(p9)
	plt.clf()
except:
	pass

print('Graphs built')
print('Done')
