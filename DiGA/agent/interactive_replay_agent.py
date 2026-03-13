
import logging
from typing import Callable, Dict, List, Optional, cast

from pandas import Timestamp

from market_simulation.states.trans_state import TransState
from market_simulation.wd.wd_order import WdOrder
from mlib.core.action import Action
from mlib.core.base_agent import BaseAgent
from mlib.core.base_order import BaseOrder
from mlib.core.observation import Observation
from mlib.core.limit_order import LimitOrder
from mlib.core.orderbook import Orderbook
from mlib.core.state import State
from mlib.core.transaction import Transaction


class ReplayAgent(BaseAgent):
    """A agent used to replay market with orders and verify with transactions."""

    def __init__(
        self,
        symbol: str,
        orders: List[BaseOrder],
        transactions: List[Transaction],
        on_order_submit: Optional[Callable[["ReplayAgent", BaseOrder], None]] = None,
    ) -> None:
        super().__init__(init_cash=0, communication_delay=0, computation_delay=0)
        self.symbol: str = symbol
        self.orders: List[BaseOrder] = orders
        self.transactions = transactions
        self._next_wakeup_order_index = 0
        self._num_check_transactions = 0
        self.on_order_submit = on_order_submit
        assert self.orders

    def get_next_wakeup_time(self, time: Timestamp) -> Optional[Timestamp]:
        if self._next_wakeup_order_index >= len(self.orders):
            return None
        next_time = self.orders[self._next_wakeup_order_index].time
        self._next_wakeup_order_index += 1
        assert next_time >= time
        return next_time

    def get_action(self, observation: Observation, orderbook: Orderbook) -> Action:
        """Get action given observation.

        It delegates its main functions to:
        - `get_next_wakeup_time` to get the next wakeup time, and
        - `get_orders` to get orders based on observation. `get_orders` will not be called for the first-time wakeup,
            when it's the market open wakeup.

        """
        assert self.agent_id == observation.agent.agent_id
        time = observation.time
        # return empty order for the market open wakeup
        orders: List[BaseOrder] = [] if observation.is_market_open_wakup else self.get_orders(time, orderbook)
        action = Action(
            agent_id=self.agent_id,
            time=time,
            orders=orders,
            next_wakeup_time=self.get_next_wakeup_time(time),
        )
        return action

    def get_orders(self, time: Timestamp, orderbook: Orderbook):
        cur_order_index = self._next_wakeup_order_index - 1
        assert cur_order_index >= 0
        order = self.orders[cur_order_index]
        assert time == order.time
        if self.on_order_submit is not None:
            self.on_order_submit(self, order)
        validated = [self.validate_order(order, orderbook)]
        return [order for order in validated if order is not None]

    def on_states_update(self, time: Timestamp, symbol_states: Dict[str, Dict[str, State]]):
        super().on_states_update(time, symbol_states)

    def check_new_transactions_match(self):
        state_name = TransState.__name__
        assert state_name in self.symbol_states[self.symbol]
        state = cast(TransState, self.symbol_states[self.symbol][state_name])
        new_trans = state.transactons[self._num_check_transactions :]
        _check_transactions_match(self.transactions, new_trans, False, self._num_check_transactions)
        self._num_check_transactions = len(state.transactons)

    def on_market_close(self, time: Timestamp):
        super().on_market_close(time)
        _check_same_symbol_orders(self.agent_id, self.lob_orders, self.lob_price_orders, self.symbol_states)

    def validate_order(self, order: WdOrder, orderbook: Orderbook):
        if order.type != 'C':
            order = order.get_limit_orders(orderbook)[0]
        else:
            valid_cancel_vol = 0
            if order.cancel_id in self.lob_orders[self.symbol].keys():
                to_cancel = self.lob_orders[self.symbol][order.cancel_id]
                valid_cancel_vol = to_cancel.volume

            if valid_cancel_vol != 0:
                order.volume = valid_cancel_vol
            else:
                logging.warning(f"Invalid order {order}.")
                order = None
        if order is not None and order.price <= 0 :
            logging.warning(f"Invalid order {order}.")
            order = None

        return order




def _check_same_symbol_orders(
    agent_id: int,
    lob_orders: Dict[str, Dict[int, LimitOrder]],
    lob_price_orders: Dict[str, Dict[int, Dict[int, LimitOrder]]],
    symbol_states: Dict[str, Dict[str, State]],
):
    symbols = lob_orders.keys()
    state_name: str = State.__name__
    for symbol in symbols:
        close_orderbook = symbol_states[symbol][state_name].close_orderbook
        if close_orderbook is None:
            # skip checking as close orderbook is empty, this happens when no close-auction.
            continue

        _check_same_orders_on_symbol(
            agent_id=agent_id,
            lob_orders=lob_orders[symbol],
            lob_price_orders=lob_price_orders[symbol],
            orderbook=close_orderbook,
        )


def _check_same_orders_on_symbol(
    agent_id: int,
    lob_orders: Dict[int, LimitOrder],
    lob_price_orders: Dict[int, Dict[int, LimitOrder]],
    orderbook: Orderbook,
):
    remaining_orders: List[LimitOrder] = []
    for level in orderbook.asks + orderbook.bids:
        remaining_orders.extend([x for x in level.orders if x.agent_id == agent_id])
    _check_same_orders(lob_orders, remaining_orders)

    price_orders: List[LimitOrder] = []
    for value in lob_price_orders.values():
        price_orders.extend(value.values())
    _check_same_orders(lob_orders, price_orders)


def _check_same_orders(lob_orders: Dict[int, LimitOrder], orders: List[LimitOrder]):
    assert len(orders) == len(lob_orders)
    for order in orders:
        assert order.order_id in lob_orders
        my_order = lob_orders[order.order_id]
        assert str(order) == str(my_order)


def _check_transactions_match(trans_label: List[Transaction], trans_replay: List[Transaction], output_details: bool = True, label_start: int = 0):
    end = label_start + len(trans_replay)
    assert label_start >= 0
    if len(trans_label) < end:
        logging.error(f"not enough transactoin [{label_start}, {end}), only {len(trans_label)}.")
        return False

    if len(trans_replay) == 0:
        return True

    for index in range(len(trans_replay)):
        str_label = str(trans_label[label_start + index])
        str_replay = str(trans_replay[index])
        if str_label == str_replay:
            if output_details:
                logging.info(f"same for {label_start + index}|{index}th trans: {str_label}")
            continue
        logging.error(f"diff for {label_start + index}|{index}th trans")
        logging.info(f"  label: {str_label}")
        logging.info(f"  reply: {str_replay}")
        return False
    return True
