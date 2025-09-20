import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(base_dir, 'data', 'processed')
input_file = os.path.join(input_dir, 'raw.csv')
data = pd.read_csv(input_file, index_col=0, parse_dates=True)

features = pd.DataFrame(index=data.index)
for col in data.columns:
    prices = data[col].fillna(method='ffill')
    features[f'{col}_ret'] = np.log(prices / prices.shift(1))
    features[f'{col}_ma5'] = prices.rolling(5).mean() / prices
    features[f'{col}_ma21'] = prices.rolling(21).mean() / prices
    features[f'{col}_vol21'] = features[f'{col}_ret'].rolling(21).std()
    features[f'{col}_rsi14'] = RSIIndicator(close=prices, window=14).rsi() / 100.0
    features[f'{col}_mom10'] = (prices - prices.shift(10)) / prices.shift(10)
    features[f'{col}_zscore21'] = (prices - prices.rolling(21).mean()) / prices.rolling(21).std()
    for fname in [f'{col}_ret', f'{col}_ma5', f'{col}_ma21', f'{col}_vol21', f'{col}_mom10', f'{col}_zscore21']:
        p1, p99 = features[fname].quantile(0.01), features[fname].quantile(0.99)
        features[fname] = features[fname].clip(p1, p99)

features = features.dropna()
features.to_csv(os.path.join(input_dir, 'features.csv'))
print('Preprocessed features data saved.')

train = features.loc[:'2017-12-31']
test = features.loc['2018-01-01':]
mean = train.mean()
std = train.std()
train_norm = (train - mean) / std
test_norm = (test - mean) / std
train_norm.to_csv(os.path.join(input_dir, 'train_norm.csv'))
test_norm.to_csv(os.path.join(input_dir, 'test_norm.csv'))
print('Training and testing data saved (normalized).')

ret_cols = [col for col in features.columns if col.endswith('_ret')]
np.save(os.path.join(input_dir, 'train_ret_mean.npy'), mean[ret_cols].values)
np.save(os.path.join(input_dir, 'train_ret_std.npy'), std[ret_cols].values)
print('Saved train_ret_mean.npy and train_ret_std.npy for model result denormalization.')
