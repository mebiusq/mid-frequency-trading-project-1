import yfinance as yf
import os
from datetime import datetime


def get_data(cfg):
    test_end = datetime.today().strftime('%Y-%m-%d')

    raw = yf.download(cfg['tickers'], start=cfg['train_start'], end=test_end, interval=cfg['timeframe'], progress=False,
                      threads=True)

    data = raw.get('Adj Close', raw.get('Close', raw))
    data = data.ffill().dropna()  # рынки FX не торгуются в выходные, а криптовалюты торгуются

    train_data = data.loc[cfg['train_start']:cfg['train_end']]
    test_data = data.loc[cfg['test_start']:]

    os.makedirs('data', exist_ok=True)
    train_data.to_csv('data/train_data.csv')
    test_data.to_csv('data/backtest_data.csv')
