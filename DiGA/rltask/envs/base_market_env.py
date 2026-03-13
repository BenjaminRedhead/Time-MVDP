from abc import abstractmethod
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Generator

from agent.rl_agent import RLAgent
from mlib.core.env import Env
from mlib.core.observation import Observation


class BaseMarketEnv(gym.Env):
    """Custom Environment that follows gym interface."""
    market_env: Env = None
    rl_agent: RLAgent = None
    obs_generator: Generator[Observation, None, None] = None

    def __init__(self, config, mode='normal', show_progress=True, discrete_action=True):
        super().__init__()
        self.config = config  # each time we create a env, the config is fixed. Maybe we can modify this setting later.
        if discrete_action:
            self.action_space = spaces.Discrete(2 * 5 * 10 + 1)
        else:
            self.action_space = spaces.Box(low=0, high=1, shape=(2 * 5 * 10 + 1,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(config["window"]+40+3,))
        self.reward_mode = config['reward_mode']
        self.show_progress = show_progress

    @abstractmethod
    def prepare_trading_env(self, config):
        # must register market_env, rl_agent, obs_generator here.
        ...

    def market_loop(self):
        observation = None
        for observation in self.obs_generator:
            if observation.agent.agent_id == self.rl_agent.agent_id and not observation.is_market_open_wakup and self.rl_state.initilized:
                break
            else:
                agent_to_act = observation.agent
                action = agent_to_act.get_action(observation)
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
            raise ValueError(f"observation not handled: {observation}")

    def step(self, action):

        action = self.rl_agent.get_action(self.env_obs, action)
        self.market_env.step(action)
        observation = self.market_loop()
        reward = self.rl_agent.get_step_pnl(self.reward_mode)

        terminated = True if observation is None else False
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        info = {}
        observation = self.prepare_trading_env(self.config)
        return observation, info

    def render(self):
        ...

    def close(self):
        ...
