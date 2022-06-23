import collections
from pathlib import Path

import numpy as np
import pandas as pd

from qlib.backtest.decision import BaseTradeDecision, Order, OrderDir, OrderHelper, TradeDecisionWO, TradeRange
from qlib.backtest.executor import NestedExecutor, SimulatorExecutor
from qlib.backtest.utils import CommonInfrastructure
from qlib.config import QlibConfig
from qlib.contrib.strategy import TWAPStrategy
from qlib.rl.order_execution.from_neutrader.executor import RLNestedExecutor
from qlib.rl.order_execution.from_neutrader.strategy import DecomposedStrategy, SingleOrderStrategy
from qlib.rl.order_execution.simulator_qlib import ExchangeConfig, QlibSimulator
from qlib.strategy.base import BaseStrategy

TIME_PER_STEP = "30min"


def _top_strategy_fn(
    common_infra: CommonInfrastructure,
    order: Order,
    trade_range: TradeRange,
    instrument: str,
) -> SingleOrderStrategy:
    return SingleOrderStrategy(common_infra, order, trade_range, instrument)


def _inner_strategy_fn(
    common_infra: CommonInfrastructure,
    order: Order,
    trade_range: TradeRange,
    instrument: str,
) -> DecomposedStrategy:
    return DecomposedStrategy()


def _inner_executor_fn(common_infra: CommonInfrastructure) -> RLNestedExecutor:
    return RLNestedExecutor(
        time_per_step=TIME_PER_STEP,
        inner_strategy=TWAPStrategy(),
        inner_executor=SimulatorExecutor(
            time_per_step="1min",
            verbose=False,
            trade_type=SimulatorExecutor.TT_SERIAL,
            generate_report=False,
            common_infra=common_infra,
            track_data=True,
        ),
        common_infra=common_infra,
        track_data=True,
    )


def test():
    order_infos = [
        ("2019-03-04", 1078.644160270691, 1),
        ("2019-03-11", 32.440425872802734, 1),
        ("2019-03-25", 40.55053234100342, 0),
        ("2019-04-01", 1070.5340538024902, 0),
        ("2019-05-27", 300.0739393234253, 1),
        ("2019-06-03", 8.110106468200684, 0),
        ("2019-06-11", 0.9360466003417968, 0),
        ("2019-06-17", 794.4272003173828, 1),
        ("2019-06-24", 7.865615844726562, 0),
        ("2019-07-01", 1077.589370727539, 0),
        ("2021-01-04", 499.7846999168396, 1),
        ("2021-01-11", 14.918946266174316, 0),
        ("2021-01-18", 484.8657536506653, 0),
        ("2021-02-08", 537.0820655822754, 1),
        ("2021-02-18", 7.459473133087158, 0),
        ("2021-02-22", 7.459473133087158, 0),
        ("2021-03-01", 14.918946266174316, 1),
        ("2021-03-08", 872.7583565711975, 1),

    ]
    orders = collections.deque([
        Order(
            stock_id="",
            amount=info[1],
            direction=OrderDir(info[2]),
            start_time=pd.Timestamp(info[0]),
            end_time=pd.Timestamp(info[0]),
        )
        for info in order_infos
    ])

    # fmt: off
    simulator = QlibSimulator(
        time_per_step=TIME_PER_STEP,
        start_time="9:45",
        end_time="14:44",
        qlib_config=QlibConfig(
            {
                "provider_uri_day": Path("C:/workspace/NeuTrader/data_sample/cn/qlib_amc_1d"),
                "provider_uri_1min": Path("C:/workspace/NeuTrader/data_sample/cn/qlib_amc_1min"),
                "feature_root_dir": Path("C:/workspace/NeuTrader/data_sample/cn/qlib_amc_handler_stock"),
                "feature_columns_today": [
                    "$open", "$high", "$low", "$close", "$vwap", "$bid", "$ask", "$volume",
                    "$bidV", "$bidV1", "$bidV3", "$bidV5", "$askV", "$askV1", "$askV3", "$askV5",
                ],
                "feature_columns_yesterday": [
                    "$open_1", "$high_1", "$low_1", "$close_1", "$vwap_1", "$bid_1", "$ask_1", "$volume_1",
                    "$bidV_1", "$bidV1_1", "$bidV3_1", "$bidV5_1", "$askV_1", "$askV1_1", "$askV3_1", "$askV5_1",
                ],
            }
        ),
        top_strategy_fn=_top_strategy_fn,
        inner_strategy_fn=_inner_strategy_fn,
        inner_executor_fn=_inner_executor_fn,
        exchange_config=ExchangeConfig(
            limit_threshold=('$ask == 0', '$bid == 0'),
            deal_price=('If($ask == 0, $bid, $ask)', 'If($bid == 0, $ask, $bid)'),
            volume_threshold={
                'all': ('cum', "0.2 * DayCumsum($volume, '9:45', '14:44')"),
                'buy': ('current', '$askV1'),
                'sell': ('current', '$bidV1'),
            },
            open_cost=0.0005,
            close_cost=0.0015,
            min_cost=5.0,
            trade_unit=100.0,
            cash_limit=None,
            generate_report=False
        ),
    )
    # fmt: on

    simulator.reset(orders.popleft())


if __name__ == "__main__":
    test()
