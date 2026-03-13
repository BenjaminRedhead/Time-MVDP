import numpy as np
import warnings
import pandas as pd
from dateutil.parser import parse
import os

from market_simulation.wd.utils import get_all_trans
from market_simulation.wd.wd_order_read_utils import read_orders
from mlib.core.env import Env
from mlib.core.event import create_exchange_events
from mlib.core.exchange import Exchange

from agent.interactive_replay_agent import ReplayAgent
from agent.rl_agent import RLAgent
from agent.utils.rl_state import RLState
from agent.utils.trade_info_state import TradeInfoState
from rltask.envs.base_market_env import BaseMarketEnv
from utils.pkl_utils import load_pkl
from pathlib import Path

from mlib.core.exchange_config import create_a_stock_exchange_config

base_replay_config = {
    "replay_path": "{your_data_path}",
    "window": 30,
    "tick_size": 100,
    "start_time": "09:30:00",
    "end_time": "15:00:00",
    "reward_mode": "step",
    "train": True,
    "test_pnl_path": None
}


class ReplayMarketEnv(BaseMarketEnv):
    """Custom Environment that follows gym interface."""

    def __init__(self, config, mode='normal', show_progress=True, discrete_action=True):
        super().__init__(config=config, mode=mode, show_progress=show_progress, discrete_action=discrete_action)
        self.current_pair = 0
        self.all_path_pairs = None
        self.is_train = config["train"]
        self.prepare_replay_data()

    def prepare_replay_data(self):
        assert 'replay_path' in self.config, "Replay path not specified!"

        self.all_path_pairs = load_pkl(self.config['replay_path'])
        self.num_pairs = len(self.all_path_pairs)
        if self.is_train:
            np.random.shuffle(self.all_path_pairs)

    def get_next_replay_data(self):
        assert self.all_path_pairs is not None, "Path pairs not prepared!"
        if self.current_pair >= self.num_pairs:
            self.current_pair = 0
        symbol, date, order_path, trans_path = self.all_path_pairs[self.current_pair]
        wd_orders = read_orders(tran_path=trans_path, order_path=order_path, symbol=symbol)
        wd_trans = get_all_trans(trans_path, symbol=symbol)

        self.current_pair += 1
        return symbol, date, wd_orders, wd_trans

    def prepare_trading_env(self, config):
        """Run a rollout and get trade info."""
        if not self.is_train and self.config['test_pnl_path'] is not None:
            if self.rl_agent is not None:
                if not os.path.exists(self.config['test_pnl_path']):
                    os.makedirs(self.config['test_pnl_path'])
                self.rl_agent.pnl.to_csv(Path(self.config['test_pnl_path'])/f"test_pnl_{self.current_pair-1}.csv")
        symbol, date, this_orders, this_trans = self.get_next_replay_data()
        self.symbol = symbol
        for trans in this_trans:
            if trans.price != 0:
                init_price = trans.price
                break
        assert init_price > 0, "Init price invalid!"
        date = pd.to_datetime(date)
        mkt_open: pd.Timestamp = date + pd.to_timedelta(parse(config["start_time"]).strftime("%H:%M:%S"))  # type: ignore
        mkt_close: pd.Timestamp = date + pd.to_timedelta(parse(config["end_time"]).strftime("%H:%M:%S"))
        self.close_time = mkt_close
        print(
            f"Reset exchange and environments...Symbol {symbol}, Data {date}"
        )
        ex_config = create_a_stock_exchange_config(date, [symbol])
        self.exchange = Exchange(ex_config)
        self.exchange.register_state(TradeInfoState())  
        self.rl_state = RLState(config['window'], init_price, config['tick_size'])
        self.exchange.register_state(self.rl_state)

        self.market_env = Env(self.exchange, "Replay Market Env For RL", show_progress=self.show_progress)
        background_agents = [
            ReplayAgent(symbol=symbol, orders=this_orders, transactions=this_trans)
        ]
        for bg_agent in background_agents:
            self.market_env.register_agent(bg_agent)
        self.rl_agent = RLAgent(start_time=mkt_open, end_time=mkt_close, obs_state=self.rl_state, symbol=symbol)
        self.market_env.register_agent(self.rl_agent)
        self.market_env.push_events(create_exchange_events(ex_config))
        self.obs_generator = self.market_env.env()
        observation = self.market_loop()

        return observation

    def market_loop(self):
        observation = None
        for observation in self.obs_generator:
            if observation.agent.agent_id == self.rl_agent.agent_id and not observation.is_market_open_wakup and self.rl_state.initilized:
                break
            else:
                agent_to_act = observation.agent
                action = agent_to_act.get_action(observation, self.exchange._orderbooks[self.symbol])
                self.market_env.step(action)
        if observation is None:
            return None
        elif isinstance(observation.agent, RLAgent):
            self.env_obs = observation
            return observation.agent.convert_state()
        elif observation.time >= self.close_time:
            return None
        elif len(self.market_env.events) == 0:
            return None
        else:
            warnings.warn(f"observation not handled: {observation}")
            return None
