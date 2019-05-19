import time
import json
import urllib, http.client
import hmac, hashlib

from urllib.parse import urlparse, urlencode


class Poloniex():
    public_methods = [
        'returnTicker',
        'return24hVolume',
        'returnOrderBook',
        'returnTradeHistory',
        'returnChartData',
        'returnCurrencies',
        'returnLoanOrders'
    ]

    def __init__(self, API_KEY, API_SECRET):
        self.API_KEY = API_KEY
        self.API_SECRET = bytearray(API_SECRET, encoding='utf-8')

    def __getattr__(self, name):
        def wrapper(*args, **kwargs):
            method = 'public' if name in self.public_methods else 'tradingApi'
            kwargs.update(method=method, command=name)
            return self.call_api(**kwargs)

        return wrapper

    def call_api(self, **kwargs):
        api_url = 'https://poloniex.com/' + kwargs['method']

        if kwargs['method'] == 'public':
            api_url += '?' + urlencode(kwargs)
            http_method = "GET"
        else:
            http_method = "POST"

        time.sleep(0.2)  # По правилам биржи нельзя больше 6 запросов в секунду
        payload = {'nonce': int(round(time.time() * 1000))}

        if kwargs:
            payload.update(kwargs)

        payload = urllib.parse.urlencode(payload)

        H = hmac.new(key=self.API_SECRET, digestmod=hashlib.sha512)
        H.update(payload.encode('utf-8'))
        sign = H.hexdigest()

        headers = {"Content-type": "application/x-www-form-urlencoded",
                   "Key": self.API_KEY,
                   "Sign": sign}

        url_o = urlparse(api_url)
        conn = http.client.HTTPSConnection(url_o.netloc)
        conn.request(http_method, api_url, payload, headers)
        response = conn.getresponse().read()

        conn.close()

        try:
            obj = json.loads(response.decode('utf-8'))

            if 'error' in obj and obj['error']:
                raise Exception(obj['error'])
            return obj
        except ValueError:
            raise Exception(
                'Получены некорректные данные (проверьте, правильно ли указан метод API {api_method})'.format(
                    api_method=kwargs['command']))