# import json
import numpy as np
import pandas as pd


class CtrlLoader:
    """Generation data loader. Load fundamentals and num of orders in terms of gt."""

    def __init__(self, duration, ctrls, symbol="000001"):
        self.symbol = symbol
        self.mkt_open = duration[0]
        self.mkt_close = duration[1]
        self.date = str(self.mkt_open.year) + "{:02d}".format(self.mkt_open.month) + "{:02d}".format(self.mkt_open.day)  # str(self.mkt_open.day)

        self.ctrls = ctrls
        if 'fundamental' in self.ctrls.keys():
            self.load_fdmtl(self.ctrls['fundamental'])
        if 'n_orders' in self.ctrls.keys():
            self.load_n_orders(self.ctrls['n_orders'])


    def load_fdmtl(self, fdmtl):
        # fdmtl: 1-d array of MINUTE mid price as fundamental, transform into second-wise df
        mkt_open, mkt_close = self.mkt_open, self.mkt_close
        date_range = pd.date_range(mkt_open, mkt_close, inclusive="left", freq='1s', name='ts')
        price_df = pd.DataFrame(index=date_range).reset_index(names=['ts'])
        sec_trading_filter = price_df['ts'].apply(lambda x: not ((x.hour==11 and x.minute>=30) or x.hour==12))
        min_date_range = pd.date_range(mkt_open, mkt_close, inclusive="left", freq='1min', name='ts')
        min_price_df = pd.DataFrame(index=min_date_range).reset_index(names=['ts'])
        min_trading_filter = min_price_df['ts'].apply(lambda x: not ((x.hour==11 and x.minute>=30) or x.hour==12))


        trading_min = min_price_df[min_trading_filter].copy()
        trading_sec = price_df[sec_trading_filter].copy()
        row_prices = fdmtl // 50 * 50
        ffill_row = np.pad(row_prices, (0, trading_min.shape[0]-row_prices.shape[0]), 'constant', constant_values=row_prices[-1])
        trading_min['price'] = ffill_row
        trading_sec = trading_sec.merge(trading_min, how='left', on='ts').ffill()
        self._midPriceTb = trading_sec

    def load_n_orders(self, n):
        mkt_open, mkt_close = self.mkt_open, self.mkt_close        
        min_date_range = pd.date_range(mkt_open, mkt_close, inclusive="left", freq='1min', name='ts')
        min_price_df = pd.DataFrame(index=min_date_range).reset_index(names=['ts'])
        min_trading_filter = min_price_df['ts'].apply(lambda x: not ((x.hour==11 and x.minute>=30) or x.hour==12))

        trading_min = min_price_df[min_trading_filter].copy()
        # should have correct shape or else will be padded with 0 and maybe raise error
        ffill_row = np.pad(n, (0, trading_min.shape[0]-n.shape[0]), 'constant', constant_values=0)
        ffill_row = np.maximum(ffill_row, 100)
        trading_min['n_orders'] = ffill_row
        self._nOrdersTb = trading_min

    def getMidPrice(self):
        return self._midPriceTb

    def getNOrders(self):
        return self._nOrdersTb

    def generateSeed(self):
        return int(self._midPriceTb['price'].mean() // 10)

