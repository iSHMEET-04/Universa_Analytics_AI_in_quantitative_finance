import yfinance as yf
import pandas as pd
from datetime import datetime
import os 

tickers= {
    'S&P500': '^GSPC',
    'FTSE100': '^FTSE',
    'Nikkei225': '^N225',
    'EEM': 'EEM',
    'Gold': 'GLD',
    'UST10Y': '^TNX'
}

start, end= '2010-01-01', '2020-12-31'
prices= pd.DataFrame()

for name, ticker in tickers.items():
    
    df= yf.download(ticker, start= start, end=end)
    print(df.columns)

    prices[name]= df['Close']

prices.dropna(inplace=True)

base_dir= os.path.dirname(os.path.abspath(__file__))
processed_dir= os.path.join(base_dir, 'processed')
filename= os.path.join(processed_dir, 'raw.csv')
prices.to_csv(filename)
print('Data Saved :)')