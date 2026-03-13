from math import ceil
from typing import List, NamedTuple, Optional, Dict, Any
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.random import RandomState
from pandas import Timestamp
from rich import print
import logging
from agent.utils.meta_oracle import CtrlOracle
from agent.utils.ctrl_loader import CtrlLoader
from agent.utils.chartist_state import ChartistState
from mlib.core.base_agent import BaseAgent
from mlib.core.base_order import BaseOrder
from mlib.core.limit_order import LimitOrder
from mlib.core.observation import Observation
from mlib.core.action import Action

base_config = {
    "f_value": "g",
    "init_price": 100000,
    "price_std": 20,
    "srcDataPth": "./data/",
    "date": "20000101",
    "start_time": "09:30:00",
    "end_time": "15:00:00",
    "tick_size": 100.0,
    "num_agent": 400,
    "wakeup_prob": 0.01,
    "delta_time": 1,
    "freq": 1,
    "f": 10,
    "c": 1.5,
    "n": 1,
    "time_ref": 30,
    "init_cash": 10,
    "init_hold": 10,
    "a": 0.1,
    "risk_averse": 0.1,
    "ob": 0,
    "g_std": 0,
    "provider_wake": 0,
    "taker_wake": 0,
    "inst_ratio": 0.11751711399421289,
    "aggressive_beta": 0,
    "marketmaker_prob": 0.4,
    "marketmaker_price_delta_limit": 99999,
    "ins_ratio": 0.11751711399421289,
    'mid_price_path': 'ctrl_diffusion_configs'
}

def get_oracle(agent_config: Dict[str, Any], ctrls: Dict[str, Any], symbol: str, output_dir: Path, mkt_open: pd.Timestamp, mkt_close: pd.Timestamp):
    data_loader = CtrlLoader(duration=[mkt_open, mkt_close], ctrls=ctrls, symbol=symbol)
    symbols = {
        symbol: {
            "a": agent_config["a"],
            "freq": agent_config["freq"],
            "dataLoader": data_loader,
            "save_pth": output_dir,
        }
    }
    oracle: CtrlOracle = CtrlOracle(mkt_open, mkt_close, symbols)
    return oracle, data_loader

def parse_config(agent_config: Dict[str, Any], symbol: str, mkt_open: pd.Timestamp, mkt_close: pd.Timestamp):
    config = CtrlAgentConfig(
        starting_cash=agent_config["init_cash"],
        starting_hold=agent_config["init_hold"],
        time_horizon_ref=agent_config["time_ref"],
        fundamentalist_rate=agent_config["f"],
        chartist_rate=agent_config["c"],
        noise_rate=agent_config["n"],
        tick_size=agent_config["tick_size"],
        risk_aversion_alpha=agent_config["risk_averse"],
        delta_time=agent_config["delta_time"],
        init_price=agent_config["init_price"],
        symbol=symbol,
        start_time=mkt_open,
        end_time=mkt_close
    )
    return config

class CtrlAgentConfig(NamedTuple):
    """Agent config for `CtrlHeterogeneousAgent`."""

    start_time: Timestamp
    end_time: Timestamp
    starting_cash: int = 10
    starting_hold: int = 10
    fundamentalist_rate: float = 10.0
    chartist_rate: float = 2.0
    noise_rate: float = 1.0
    tick_size: int = 100
    time_horizon_ref: float = 30
    risk_aversion_alpha: float = 0.1
    delta_time: float = 1  # in seconds
    noise_var: float = 1e-4
    keep_size_prob: float = 1  
    min_num_returns: int = 20
    init_price: int = 300  # a pesudo price for the agent
    symbol: str = "symbol_name"
    MIN_RETURN_VAR: float = 1.0e-10

def in_noon_break(current_time: Timestamp):
    """To check if current time is in the noon break."""
    if current_time.hour == 11 and current_time.minute >= 30 or current_time.hour == 12:
        return True
    else:
        return False

class CtrlHeterogeneousAgent(BaseAgent):
    """The heterogeneous finance agent."""

    def __init__(
        self,
        agent_id: int,
        config: CtrlAgentConfig,
        chartist_state: ChartistState,
        oracle: CtrlOracle,
        symbol: str = "symbol_name",
    ):
        super().__init__()
        self.agent_id: int = agent_id
        self.config = config
        self.noise_var: float = config.noise_var
        self.time_horizon_ref: float = config.time_horizon_ref
        self.risk_aversion_alpha: float = config.risk_aversion_alpha
        self.f = config.fundamentalist_rate
        self.c = config.chartist_rate
        self.n = config.noise_rate
        self.delta_time: float = config.delta_time  # in second

        self.starting_cash: int = config.starting_cash
        self.starting_hold: int = config.starting_hold
        self.random_state: RandomState = RandomState(seed=np.random.randint(low=0, high=2**32, dtype="uint64"))  # type: ignore
        self.symbol = symbol
        self.price_scale: float = config.init_price / 300

        self.init_price = config.init_price
        self.current_price: float = config.init_price / self.price_scale
        self.pre_price: float = config.init_price / self.price_scale
        self.tick_size: float = config.tick_size

        self.estimated_price: float = config.init_price
        self.return_var: float = 0.0

        self.chartist_state = chartist_state
        self.chartist_state.register_horizon(2000)
        self.min_num_returns = config.min_num_returns
        self.keep_size_prob: float = config.keep_size_prob

        self.start_trading_time: Optional[pd.Timestamp] = None
        self.next_wakeup_time = None
        self.MIN_RETURN_VAR: float = config.MIN_RETURN_VAR

        self.oracle = oracle
        self.wakeup_seconds = self.oracle.scheduleWakeUpTime(symbol)
        print(f"Scuduled number of wakeups: {len(self.wakeup_seconds)}")
        self.volume_scale = 100
        self.order_horizon = {}  # {order_id: horizon} pairs

    def agent_init(self, f: float, c: float, n: float):  # initialize a "new" agent for each wakeup
        self.fundamentalist_rate = abs(np.random.laplace(scale=f))
        self.chartist_rate = abs(np.random.laplace(scale=c))
        self.noise_rate = abs(np.random.laplace(scale=n))
        self.shadow_position: int = int(self.random_state.exponential(self.starting_hold)) + 1  # type: ignore
        self.shadow_cash = self.shadow_position * 300.0

        self.fundamentalist_tau: int = ceil(
            self.time_horizon_ref * (1 + self.fundamentalist_rate)
        )  
        self.time_horizon: int = min(ceil(self.time_horizon_ref * (1 + self.fundamentalist_rate) / (1 + self.chartist_rate)), 2000)  # formula (4) in ref paper
        self.risk_aversion: float = self.risk_aversion_alpha * (1 + self.fundamentalist_rate) / (1 + self.chartist_rate)  # formula (6) in ref paper

        self.cash = self.tradable_cash = self.shadow_cash * self.price_scale * self.volume_scale
        self.holdings[self.symbol] = self.tradable_holdings[self.symbol] = self.shadow_position * self.volume_scale

    def on_market_open(self, time: Timestamp, symbols: List[str]) -> None:
        super().on_market_open(time, symbols)

    def on_order_accepted(self, time: Timestamp, orders: List[LimitOrder]) -> None:
        for order in orders:
            self._add_accepted_order(order)
            self.order_horizon[order.order_id] = self.time_horizon

    def get_action(self, observation: Observation) -> Action:

        assert self.agent_id == observation.agent.agent_id
        time = observation.time
        # return empty order for the market open wakeup
        orders: List[BaseOrder] = [] if observation.is_market_open_wakup else self.get_orders(time)
        action = Action(
            agent_id=self.agent_id,
            time=time,
            orders=orders,
            next_wakeup_time=self.get_next_wakeup_time(time),
        )
        return action

    def get_next_wakeup_time(self, time: Timestamp) -> Optional[Timestamp]:
        if not self.next_wakeup_time and self.wakeup_seconds:
            self.next_wakeup_time = self.wakeup_seconds[0]
            self.wakeup_seconds.pop(0)
            return self.next_wakeup_time
        if self.next_wakeup_time > time:  # not yet time to wakeup yet, the correct wake up event is not handled, so don't push a new wakeup event
            return None
        if self.wakeup_seconds:
            self.next_wakeup_time = self.wakeup_seconds[0]
            self.wakeup_seconds.pop(0)
            return self.next_wakeup_time
        return None

    def get_orders(self, time: Timestamp) -> List[BaseOrder]:
        self.agent_init(self.f, self.c, self.n)
        self.current_price = self.chartist_state.current_price
        self.pre_price = self.chartist_state.pre_price
        self.shadow_position = self.tradable_holdings[self.symbol] / self.volume_scale
        self.shadow_cash = self.tradable_cash / (self.price_scale * self.volume_scale)
        cancel_orders = self.get_expired_orders_for_cancel(time)
        limit_orders = self._get_orders(current_time=time)
        orders: List[BaseOrder] = cancel_orders + limit_orders  # type: ignore

        return orders

    def get_expired_orders_for_cancel(self, current_time: pd.Timestamp):
        orders: List[LimitOrder] = []
        orders_to_cancel: List[LimitOrder] = []
        for order in self.lob_orders[self.symbol].values():
            horizon = self.order_horizon[order.order_id]
            cancel_time = order.time + pd.Timedelta(value=int(round(self.delta_time * horizon)), unit="S")
            if cancel_time < current_time:
                orders_to_cancel.append(order)

        for _, order in enumerate(orders_to_cancel):
            orders.append(self.put_cancel_order(dt=current_time, order=order))

        return orders

    def _get_orders(self, current_time: pd.Timestamp):
        orders: List[LimitOrder] = []
        if self.chartist_state.is_ready():
            self.estimated_price = self.update_estimated_price(current_time)
            self.return_var = max(self.MIN_RETURN_VAR, self.chartist_state.cal_return_variance(self.time_horizon))
            orders = self.place_order(current_time)
        else:
            # update price history from oracle agent
            obs_t: int = self.oracle.observePrice(self.symbol, current_time, random_state=self.random_state)  # type: ignore
            self.current_price = obs_t / self.price_scale + self.random_state.normal(loc=0, scale=1e-3)
            self.chartist_state.observe_price(self.current_price, current_time)
        return orders

    def update_estimated_chartist(self):
        avg_return: float = self.chartist_state.estimate_return(self.time_horizon)
        return avg_return

    def update_noise(self):
        return self.random_state.normal(loc=0, scale=self.noise_var)

    def update_estimated_return(self, current_time: pd.Timestamp):
        fundamentist: float = self.update_estimated_fundamentist(current_time)
        chartist: float = self.update_estimated_chartist()
        noise: float = self.update_noise()

        # formula (1) in ref paper
        estimated_return: float = (self.fundamentalist_rate * fundamentist + self.chartist_rate * chartist + self.noise_rate * noise) / (
            self.fundamentalist_rate + self.chartist_rate + self.noise_rate
        )
        assert estimated_return is not None
        self.chartist_state.check_valid_return(estimated_return)
        return estimated_return

    def update_estimated_price(self, current_time: pd.Timestamp):
        estimated_return: float = self.update_estimated_return(current_time)
        # formula (3) in ref paper
        estimated_price: float = self.current_price * np.exp(estimated_return * self.time_horizon)
        return estimated_price

    def cal_expected_holdNum(self, price: float):
        assert self.return_var > 0
        # formula (10) in ref paper
        expected_hold_num: float = np.log(self.estimated_price / price) / (self.risk_aversion * self.return_var * price)
        assert not np.isinf(expected_hold_num)
        return expected_hold_num

    def update_estimated_fundamentist(self, current_time: pd.Timestamp):
        obs_t: int = self.oracle.observePrice(self.symbol, current_time, random_state=self.random_state) / self.price_scale  # type: ignore
        r_T: float = np.log(obs_t / self.current_price) / self.fundamentalist_tau
        return r_T

    def cal_lowest_price(self):
        low_price = 0
        high_price = self.estimated_price
        lowest_price = high_price
        expected_hold_num = self.cal_expected_holdNum(lowest_price)
        assert expected_hold_num is not None
        cash_left = self.shadow_cash - lowest_price * (expected_hold_num - self.shadow_position)
        max_iter = 10000
        for itr in range(max_iter):
            lowest_price = (high_price + low_price) / 2
            expected_hold_num = self.cal_expected_holdNum(lowest_price)
            assert expected_hold_num is not None
            assert not np.isinf(expected_hold_num)
            cash_left = self.shadow_cash - lowest_price * (expected_hold_num - self.shadow_position)
            if 0 < cash_left < 1:
                break
            if cash_left > 0:
                high_price = lowest_price
            else:
                low_price = lowest_price
        if itr == max_iter - 1:
            lowest_price = high_price
        return lowest_price

    def sample_price_uniform(self):
        lowest_price = self.cal_lowest_price()
        self.lowest_price = lowest_price
        price = self.random_state.uniform(lowest_price, self.estimated_price)
        expected_hold_num = self.cal_expected_holdNum(price)
        return price, expected_hold_num

    def get_wakeup_prob(self, current_time: pd.Timestamp):
        return self.oracle.observeWakeupProb(self.symbol,current_time,self.config.num_agent)

    def put_limit_order(self, dt: pd.Timestamp, volume: int, is_buy_order: bool, price: int):
        order = LimitOrder(
            order_id=-1,
            symbol=self.symbol,
            time=dt,
            price=price,
            volume=volume,
            type="B" if is_buy_order else "S",
            cancel_type="None",
            cancel_id=-1,
            agent_id=self.agent_id,
            tag="",
        )

        return order

    def put_cancel_order(self, dt: pd.Timestamp, order: LimitOrder):
        cancel_id = order.order_id
        order = LimitOrder(
            order_id=-1,
            symbol=self.symbol,
            time=dt,
            price=order.price,
            volume=order.volume,
            type="C",
            cancel_type=order.type,
            cancel_id=cancel_id,
            agent_id=self.agent_id,
            tag="",
        )
        return order

    def place_order(self, current_time: pd.Timestamp):
        orders: List[LimitOrder] = []
        price, expected_hold_num = self.sample_price_uniform()

        if not (price * (expected_hold_num - self.shadow_position) < self.shadow_cash):
            logging.warn(f'Out of capital, order not placed! price: {price}, expected_hold_num: {expected_hold_num}, shadow_position: {self.shadow_position}, shadow_cash: {self.shadow_cash}')
            return orders

        if price is None or expected_hold_num is None:
            print("Samples out of range. ts: {}, price: {}".format(current_time, price))
            return orders

        if price >= self.estimated_price:
            print(
                "Warning: cannot make an order. Time: {}, agent_id: {}, cash: {}, holdings: {}, price: {}, size: {}, est_price: {}, cur_price: {}".format(
                    current_time,
                    self.agent_id,
                    self.shadow_cash,
                    self.shadow_position,
                    price,
                    expected_hold_num,
                    self.estimated_price,
                    self.current_price,
                )
            )
            return orders

        size = abs(expected_hold_num - self.shadow_position)
        if size <= 0.5:
            return orders

        rescaled_price = round(price * self.price_scale / self.tick_size) * self.tick_size
        price = rescaled_price / self.price_scale
        if not (0.5 < rescaled_price / self.init_price < 2):
            logging.warn(f"Extrerme price, order not placed! Price: {rescaled_price}, init_price: {self.init_price}, low_price: {self.lowest_price}, esti_price: {self.estimated_price}, cur_price: {self.current_price}")
            return orders
        if expected_hold_num > self.shadow_position:
            # buy
            buy = True
            size = max(1, int(size))
            if size * price < self.shadow_cash:
                if size == 1 or self.random_state.uniform() < self.keep_size_prob:
                    orders.append(self.put_limit_order(dt=current_time, volume=size * self.volume_scale, is_buy_order=buy, price=round(price * self.price_scale)))
                elif size % 2 == 0:
                    for _ in range(int(size / 2)):
                        orders.append(self.put_limit_order(dt=current_time, volume=2 * self.volume_scale, is_buy_order=buy, price=round(price * self.price_scale)))
                else:
                    for _ in range(size):
                        orders.append(self.put_limit_order(dt=current_time, volume=1 * self.volume_scale, is_buy_order=buy, price=round(price * self.price_scale)))
                self.shadow_cash -= price * size
        else:
            buy = False
            size = max(1, int(size))
            if size < self.shadow_position:
                if size == 1 or self.random_state.uniform() < self.keep_size_prob:
                    orders.append(self.put_limit_order(dt=current_time, volume=size * self.volume_scale, is_buy_order=buy, price=round(price * self.price_scale)))
                elif size % 2 == 0:
                    for _ in range(int(size / 2)):
                        orders.append(self.put_limit_order(dt=current_time, volume=2 * self.volume_scale, is_buy_order=buy, price=round(price * self.price_scale)))
                else:
                    for _ in range(size):
                        orders.append(self.put_limit_order(dt=current_time, volume=1 * self.volume_scale, is_buy_order=buy, price=round(price * self.price_scale)))
                self.shadow_position -= size

        return orders
