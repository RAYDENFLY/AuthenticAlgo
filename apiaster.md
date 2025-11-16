1. Informasi Akun Futures (Balance)
Endpoint: GET /fapi/v2/balance

javascript
// Contoh request dengan HMAC SHA256 signature
const timestamp = Date.now();
const queryString = `timestamp=${timestamp}&recvWindow=5000`;
const signature = crypto.createHmac('sha256', secretKey).update(queryString).digest('hex');

// Request
fetch('https://fapi.asterdex.com/fapi/v2/balance?' + queryString + '&signature=' + signature, {
    method: 'GET',
    headers: {
        'X-MBX-APIKEY': apiKey
    }
})
Response:

json
[
    {
        "accountAlias": "SgsR",
        "asset": "USDT",
        "balance": "122607.35137903",
        "crossWalletBalance": "23.72469206",
        "crossUnPnl": "0.00000000",
        "availableBalance": "23.72469206",
        "maxWithdrawAmount": "23.72469206",
        "marginAvailable": true,
        "updateTime": 1617939110373
    }
]
2. Informasi Akun Lengkap
Endpoint: GET /fapi/v4/account

javascript
const timestamp = Date.now();
const queryString = `timestamp=${timestamp}&recvWindow=5000`;
const signature = crypto.createHmac('sha256', secretKey).update(queryString).digest('hex');

// Request
fetch('https://fapi.asterdex.com/fapi/v4/account?' + queryString + '&signature=' + signature, {
    method: 'GET',
    headers: {
        'X-MBX-APIKEY': apiKey
    }
})
Response: Menampilkan informasi lengkap termasuk:

Saldo semua aset

Posisi terbuka

Margin

Unrealized PnL

Leverage

3. Posisi Terbuka
Endpoint: GET /fapi/v2/positionRisk

javascript
const timestamp = Date.now();
const queryString = `timestamp=${timestamp}&recvWindow=5000`;
const signature = crypto.createHmac('sha256', secretKey).update(queryString).digest('hex');

// Request
fetch('https://fapi.asterdex.com/fapi/v2/positionRisk?' + queryString + '&signature=' + signature, {
    method: 'GET',
    headers: {
        'X-MBX-APIKEY': apiKey
    }
})
4. Order Terbuka
Endpoint: GET /fapi/v1/openOrders

javascript
// Untuk semua symbol
const timestamp = Date.now();
const queryString = `timestamp=${timestamp}&recvWindow=5000`;
const signature = crypto.createHmac('sha256', secretKey).update(queryString).digest('hex');

fetch('https://fapi.asterdex.com/fapi/v1/openOrders?' + queryString + '&signature=' + signature, {
    method: 'GET',
    headers: {
        'X-MBX-APIKEY': apiKey
    }
})
5. Riwayat Trade
Endpoint: GET /fapi/v1/userTrades

javascript
const symbol = 'BTCUSDT';
const timestamp = Date.now();
const queryString = `symbol=${symbol}&timestamp=${timestamp}&recvWindow=5000`;
const signature = crypto.createHmac('sha256', secretKey).update(queryString).digest('hex');

fetch('https://fapi.asterdex.com/fapi/v1/userTrades?' + queryString + '&signature=' + signature, {
    method: 'GET',
    headers: {
        'X-MBX-APIKEY': apiKey
    }
})
6. Income History
Endpoint: GET /fapi/v1/income

javascript
const timestamp = Date.now();
const queryString = `timestamp=${timestamp}&recvWindow=5000&limit=1000`;
const signature = crypto.createHmac('sha256', secretKey).update(queryString).digest('hex');

fetch('https://fapi.asterdex.com/fapi/v1/income?' + queryString + '&signature=' + signature, {
    method: 'GET',
    headers: {
        'X-MBX-APIKEY': apiKey
    }
})
Contoh Implementasi Python:
python
import hmac
import hashlib
import requests
import time
import json

class AsterTrading:
    def __init__(self, api_key, secret_key):
        self.base_url = "https://fapi.asterdex.com"
        self.api_key = api_key
        self.secret_key = secret_key
    
    def _generate_signature(self, params):
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        return hmac.new(
            self.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def get_account_info(self):
        params = {
            'timestamp': int(time.time() * 1000),
            'recvWindow': 5000
        }
        params['signature'] = self._generate_signature(params)
        
        headers = {'X-MBX-APIKEY': self.api_key}
        response = requests.get(
            f"{self.base_url}/fapi/v4/account",
            params=params,
            headers=headers
        )
        return response.json()
    
    def get_balance(self):
        params = {
            'timestamp': int(time.time() * 1000),
            'recvWindow': 5000
        }
        params['signature'] = self._generate_signature(params)
        
        headers = {'X-MBX-APIKEY': self.api_key}
        response = requests.get(
            f"{self.base_url}/fapi/v2/balance",
            params=params,
            headers=headers
        )
        return response.json()
    
    def get_open_positions(self):
        params = {
            'timestamp': int(time.time() * 1000),
            'recvWindow': 5000
        }
        params['signature'] = self._generate_signature(params)
        
        headers = {'X-MBX-APIKEY': self.api_key}
        response = requests.get(
            f"{self.base_url}/fapi/v2/positionRisk",
            params=params,
            headers=headers
        )
        return response.json()

# Usage
trading = AsterTrading("your_api_key", "your_secret_key")
account_info = trading.get_account_info()
balance = trading.get_balance()
positions = trading.get_open_positions()Trade Panel