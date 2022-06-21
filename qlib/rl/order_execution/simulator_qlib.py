# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Placeholder for qlib-based simulator."""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generator, List, Optional, Tuple, Union

import pandas as pd
import qlib

from qlib.backtest import get_exchange
from qlib.backtest.account import Account
from qlib.backtest.decision import Order, TradeRange, TradeRangeByTime
from qlib.backtest.executor import BaseExecutor, NestedExecutor
from qlib.backtest.utils import CommonInfrastructure
from qlib.config import QlibConfig
from qlib.rl.simulator import ActType, Simulator, StateType
from qlib.strategy.base import BaseStrategy


@dataclass
class RuntimeConfig:
    seed: int = 42
    output_dir: Optional[Path] = None
    checkpoint_dir: Optional[Path] = None
    tb_log_dir: Optional[Path] = None
    debug: bool = False
    use_cuda: bool = True


@dataclass
class ExchangeConfig:
    limit_threshold: Union[float, Tuple[str, str]]
    deal_price: Union[str, Tuple[str, str]]
    volume_threshold: dict
    open_cost: float = 0.0005
    close_cost: float = 0.0015
    min_cost: float = 5.
    trade_unit: Optional[float] = 100.
    cash_limit: Optional[Union[Path, float]] = None
    generate_report: bool = False


def get_common_infra(
    config: ExchangeConfig,
    trade_start_time: pd.Timestamp,
    trade_end_time: pd.Timestamp,
    codes: List[str],
    cash_limit: Optional[float] = None,
) -> CommonInfrastructure:
    # need to specify a range here for acceleration
    if cash_limit is None:
        trade_account = Account(
            init_cash=int(1e12),
            benchmark_config={},
            pos_type='InfPosition'
        )
    else:
        trade_account = Account(
            init_cash=cash_limit,
            benchmark_config={},
            pos_type='Position',
            position_dict={code: {"amount": 1e12, "price": 1.} for code in codes}
        )

    exchange = get_exchange(
        codes=codes,
        freq='1min',
        limit_threshold=config.limit_threshold,
        deal_price=config.deal_price,
        open_cost=config.open_cost,
        close_cost=config.close_cost,
        min_cost=config.min_cost if config.trade_unit is not None else 0,
        start_time=pd.Timestamp(trade_start_time),
        end_time=pd.Timestamp(trade_end_time) + pd.DateOffset(1),
        trade_unit=config.trade_unit,
        volume_threshold=config.volume_threshold
    )

    return CommonInfrastructure(trade_account=trade_account, trade_exchange=exchange)


class QlibSimulator(Simulator[Order, StateType, ActType]):
    def __init__(
        self,
        time_per_step: str,
        start_time: str,
        end_time: str,
        qlib_config: QlibConfig,
        top_strategy_fn: Callable[[CommonInfrastructure, Order, TradeRange, str], BaseStrategy],
        inner_strategy_fn: Callable[[CommonInfrastructure, Order, TradeRange, str], BaseStrategy],
        inner_executor_fn: Callable[[CommonInfrastructure], BaseExecutor],
        exchange_config: ExchangeConfig,
    ) -> None:
        super(QlibSimulator, self).__init__(
            initial=None,  # TODO
        )

        self._trade_range = TradeRangeByTime(start_time, end_time)
        self._qlib_config = qlib_config
        self._time_per_step = time_per_step
        self._top_strategy_fn = top_strategy_fn
        self._inner_executor_fn = inner_executor_fn
        self._inner_strategy_fn = inner_strategy_fn
        self._exchange_config = exchange_config

        self._executor: Optional[NestedExecutor] = None
        self._collect_data_loop: Optional[Generator] = None

        self._done = False

    def reset(
        self,
        order: Order,
        instrument: str = "SH600000",  # TODO: Test only. Remove this default value later.
    ) -> None:
        # TODO: init_qlib

        common_infra = get_common_infra(
            self._exchange_config,
            trade_start_time=order.start_time,
            trade_end_time=order.end_time,
            codes=[instrument],
        )

        self._executor = NestedExecutor(
            time_per_step=self._time_per_step,
            inner_executor=self._inner_executor_fn(common_infra),
            inner_strategy=self._inner_strategy_fn(common_infra, order, self._trade_range, instrument),
            track_data=True,
        )

        top_strategy = self._top_strategy_fn(common_infra, order, self._trade_range, instrument)

        self._executor.reset(start_time=order.start_time, end_time=order.end_time)
        top_strategy.reset(level_infra=self._executor.get_level_infra())
        self._collect_data_loop = self._executor.collect_data(top_strategy.generate_trade_decision(), level=0)
        assert isinstance(self._collect_data_loop, Generator)

        self._done = False

    def step(self, action: ActType) -> None:
        try:
            strategy = self._collect_data_loop.send(action)
            while not isinstance(strategy, BaseStrategy):
                strategy = self._collect_data_loop.send(action)
            assert isinstance(strategy, BaseStrategy)
        except StopIteration:
            self._done = True

    def get_state(self) -> StateType:
        pass  # TODO: Collect info from executor. Generate state.

    def done(self) -> bool:
        return self._done
