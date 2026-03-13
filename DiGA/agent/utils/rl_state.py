import pandas as pd
from mlib.core.state import State
from mlib.core.trade_info import TradeInfo
import numpy as np

def rpadton(x, v=0, n=10):
    # right pad to n
    pad_len = n - len(x)
    return np.pad(x, (0, pad_len), 'constant', constant_values=(v, v))

class RLState(State):

    def __init__(self, window=20, init_price=100000, tick_size=100) -> None:
        super().__init__()
        self.signal = 0
        self.price_history = pd.DataFrame(columns=['price'], index=pd.to_datetime([]))
        self.last_price = init_price
        self.prev_price = init_price
        self.initilized = False
        self.prev_update_time = None
        self.window = window
        self.tick_size = tick_size

    def on_open(self, trade_info: TradeInfo=None, cancel_transactions = None, lob_snapshot = None, match_trans = None):
        if trade_info is not None:
            self.update_price_history()
            self.update_orderbook(trade_info.lob_snapshot)

    def on_trading(self, trade_info: TradeInfo):
        super().on_trading(trade_info)
        self.update_price_history()
        self.update_orderbook(trade_info.lob_snapshot)
        self.initilized = True

    def update_price_history(self):
        now_second = self.time.ceil('s')
        if self.prev_update_time is None:
            self.price_history.loc[now_second] = self.last_price
        else:
            prev_second = self.prev_update_time.ceil('s')
            if now_second == prev_second:
                self.price_history.loc[now_second] = self.last_price
            else:
                delta_second = now_second - prev_second
                num_of_intervals = int(delta_second.total_seconds() / 60)
                if num_of_intervals > 1:
                    for i in range(1, num_of_intervals):
                        self.price_history.loc[prev_second + pd.Timedelta(seconds=i)] = self.prev_price

                self.price_history.loc[now_second] = self.last_price

        self.prev_update_time = self.time
        self.prev_price = self.last_price

    def update_orderbook(self, orderbook):
        self.bid_prices = orderbook.bid_prices[:10]
        self.ask_prices = orderbook.ask_prices[:10]
        self.bid_volumes = orderbook.bid_volumes[:10]
        self.ask_volumes = orderbook.ask_volumes[:10]

    def get_state(self):
        # right pad to n
        history = rpadton(np.log(self.price_history.iloc[-self.window:].values / self.last_price).squeeze(-1), n=self.window)
        lob_prices = np.log(np.concatenate([rpadton(self.bid_prices, self.last_price),rpadton(self.ask_prices, self.last_price)])) / np.log(self.last_price)
        lob_volumes = (np.array(np.concatenate([rpadton(self.bid_volumes),rpadton(self.ask_volumes)]))-1000)/1000
        return np.concatenate([history, lob_prices, lob_volumes], axis=0)


