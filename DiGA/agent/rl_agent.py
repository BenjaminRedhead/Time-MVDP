from typing import List

import numpy as np
import pandas as pd
from pandas import Timestamp
# from mlib.core.state import State
from mlib.core.action import Action
from mlib.core.base_agent import BaseAgent
from mlib.core.base_order import BaseOrder
from mlib.core.lob_snapshot import LobSnapshot
from mlib.core.observation import Observation
from agent.utils.rl_state import RLState

class RLAgent(BaseAgent):  # this agent is for interacting with market environment
    """An agent used for RL algorithms."""
    def __init__(
        self,
        start_time: Timestamp,
        end_time: Timestamp,
        obs_state: RLState,
        symbol: str = "000001",
        init_cash: float = 1e8,
        communication_delay: int = 0,
        computation_delay: int = 0,
        trade_period: str = '10s'
    ) -> None:
        super().__init__(init_cash, communication_delay, computation_delay)
        self.init_cash = init_cash
        self.capital = init_cash
        self.last_capital = init_cash
        self.symbol = symbol
        self.obs_state = obs_state
        self.pnl = pd.DataFrame(columns=['capital', 'position', 'cash'], index=pd.to_datetime([]))
        self.start_time = start_time
        self.end_time = end_time
        self.trade_period = trade_period

    def get_action(self, observation: Observation, action=None) -> Action:
        assert self.agent_id == observation.agent.agent_id
        # return empty order for the market open wakeup
        time = observation.time
        if action is not None:
            action = self.convert_action(action, time)
        else:
            orders: List[BaseOrder] = []
            action = Action(
                agent_id=self.agent_id,
                time=time,
                orders=orders,
                next_wakeup_time=self.get_next_wakeup_time(time),
            )
        return action

    def convert_action(self, action, time: Timestamp):
        orders: List[BaseOrder] = self.get_orders(time, action)
        env_action = Action(
            agent_id=self.agent_id,
            time=time,
            orders=orders,
            next_wakeup_time=self.get_next_wakeup_time(time),
        )
        return env_action

    def convert_state(self):
        time = self.obs_state.time
        self.position = self.holdings[self.symbol]
        self.capital = self.position * self.obs_state.last_price + self.cash
        self.step_pnl = self.capital - self.last_capital
        self.last_capital = self.capital
        self.pnl_state = [self.capital, self.position, self.cash]
        self.pnl.loc[time] = self.pnl_state

        env_state = self.obs_state.get_state()
        self_state = np.array(self.pnl_state) / self.init_cash
        concat_state = np.concatenate([env_state, self_state]).astype(np.float32)
        return concat_state

    def get_orders(self, time, action):
        orders: List[BaseOrder] = []
        lob: LobSnapshot = self.obs_state.lob_snapshot
        if isinstance(action, (int, np.int_)):
            act_idx = action
        else:
            act_idx: int = np.argmax(action)
        if act_idx == 0:
            return orders
        if (act_idx - 1) // 50 == 0:
            order_type = 'B'
        else:
            order_type = 'S'

        type_act = (act_idx - 1) % 50
        price_slot = type_act // 10
        volume = int(type_act % 10 + 1) * 100

        if order_type == 'B':
            price = lob.ask_prices[price_slot] if len(lob.ask_prices) > price_slot else self.obs_state.last_price + price_slot * self.obs_state.tick_size
            if price * volume < self.cash:
                orders = self.construct_valid_orders(time, self.symbol, order_type, price, volume)
        else:
            if self.position >= volume:
                price = lob.bid_prices[price_slot] if len(lob.bid_prices) > price_slot else self.obs_state.last_price
                orders = self.construct_valid_orders(time, self.symbol, order_type, price, volume)

        return orders

    def get_next_trading_second(self, time: Timestamp, period='1s'):
        next_time = time + pd.Timedelta(period)
        if next_time.hour >= 15:
            return None
        while (next_time.hour == 11 and next_time.minute >= 30) or next_time.hour == 12:
            next_time += pd.Timedelta(period)
        return next_time

    def get_next_trading_minute(self, time: Timestamp):
        next_time = time + pd.Timedelta(minutes=1)
        if next_time.hour >= 15:
            return None
        while (next_time.hour == 11 and next_time.minute >= 30) or next_time.hour == 12:
            next_time += pd.Timedelta(minutes=1)
        return next_time

    def get_next_wakeup_time(self, time: Timestamp):
        '''
            Wake up every second.
        '''
        return self.get_next_trading_second(time, self.trade_period)

    def get_pnl(self):
        return self.pnl

    def get_step_pnl(self, mode='step'):
        if mode == 'step':
            return self.step_pnl / self.init_cash * 10000
        elif mode == 'acc':
            return (self.capital - self.init_cash) / self.init_cash * 10000
        else:
            raise NotImplementedError(f"Reward mode {mode} is not implemented.")
