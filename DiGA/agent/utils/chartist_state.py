import random
from typing import List, Optional

import numpy as np
import pandas as pd
from pandas import Timestamp

from mlib.core.state import State
from mlib.core.trade_info import TradeInfo


class ChartistState(State):
    def __init__(self, delta_time: float, init_len: int, init_price: float) -> None:
        super().__init__()
        self.delta_time = delta_time
        self.init_len = init_len

        self.horizon = 0
        self.update_time: Optional[Timestamp] = None
        self.return_history = pd.DataFrame(columns=["return"], index=pd.to_datetime([]))  # type: ignore
        self._is_ready = False

        self._return_vars: List[float] = []
        self._time_deltas: List[float] = []
        self._num_returns = 0
        self._num_returns_by_1s = 0

        self.price_scale = init_price / 300
        self.pre_price: float = init_price / self.price_scale
        self.current_price: float = init_price / self.price_scale
        self.last_trade_info: Optional[TradeInfo] = None

    def on_trading(self, trade_info: TradeInfo):
        super().on_trading(trade_info)
        self.last_trade_info = trade_info

        self.pre_price = self.current_price
        self.current_price = self.get_mid_price()
        ret = self.get_current_return()
        self.update_historical_data(ret, time=trade_info.order.time)

    def observe_price(self, price: float, time: Timestamp):
        self.current_price = price
        ret = self.get_current_return()
        self.pre_price = self.current_price
        self.update_historical_data(ret, time=time)

    def check_valid_return(self, ret: float):
        assert -0.5 < ret < 0.5  # if error encountered, you can temporarilly disabled this assertion
        pass

    def get_mid_price(self):
        if self.last_trade_info is None:
            return self.pre_price
        lob_snapshot = self.last_trade_info.lob_snapshot
        if lob_snapshot.ask_prices and lob_snapshot.bid_prices:
            mid_price = (lob_snapshot.ask_prices[0] + lob_snapshot.bid_prices[0]) / 2
            mid_price /= self.price_scale
            return mid_price
        return self.pre_price

    def get_current_return(self):
        if self.current_price is None or self.pre_price is None:
            return 0
        assert self.current_price > 0
        assert self.pre_price > 0
        # formula (2) in reg paper
        current_return: float = np.log(self.current_price / self.pre_price)
        assert current_return is not None
        return current_return

    def is_ready(self):
        return self._is_ready

    def estimate_return(self, horizon: int):
        avg_return: float = self.return_history[-int(horizon) :]["return"].mean()  # type: ignore
        self.check_valid_return(avg_return)
        return avg_return

    def cal_return_variance(self, horizon: int):
        avg_return: float = self.estimate_return(horizon)
        var_return = sum(map(lambda x: (x - avg_return) ** 2, self.return_history[-int(horizon) :]["return"].tolist())) / horizon  # type: ignore
        self._return_vars.append(var_return)
        return var_return

    # Return and Time
    def update_historical_data(self, x: float, time: Timestamp):
        self.check_valid_return(x)
        # Initialization
        if not self.update_time:
            self.return_history = pd.concat([self.return_history, pd.DataFrame([pd.Series({"return": x}, name=time)])]) # self.return_history.append(pd.Series({"return": x}, name=time))  # type: ignore
            self.return_history.dropna(inplace=True)  # type: ignore
            self.update_time = time
            return

        time_delta_from_last_msg = int((time - self.update_time).total_seconds() / self.delta_time)
        assert time_delta_from_last_msg >= 0

        if time_delta_from_last_msg == 0:
            return
        elif time_delta_from_last_msg > 1:
            self._time_deltas.append(time_delta_from_last_msg)
            # Time interpolation
            for idx_delta in range(time_delta_from_last_msg - 1):
                self.return_history = pd.concat([self.return_history, pd.DataFrame([pd.Series({"return": 0}, name=self.update_time + pd.Timedelta("{}s".format(self.delta_time * idx_delta)))])])
                self._num_returns_by_1s += 1
        self._num_returns += 1
        # update return
        self.return_history = pd.concat([self.return_history, pd.DataFrame([pd.Series({"return": x}, name=time)])]) # self.return_history.append(pd.Series({"return": x}, name=time))  # type: ignore
        self.return_history.dropna(inplace=True)  # type: ignore
        self.update_time = time

        self._clear_queue()
        self._is_ready = True if len(self.return_history) > self.init_len else False  # type: ignore

    def register_horizon(self, x: int) -> None:
        x = int(x)
        if x > self.horizon:
            self.horizon = x
            # print("Maximum chartist horizon: {}".format(self.horizon))
        return

    # prevent the queue from growing too long
    def _clear_queue(self):
        if random.random() < 0.1:
            self.return_history = self.return_history[-self.horizon - 1 :]
