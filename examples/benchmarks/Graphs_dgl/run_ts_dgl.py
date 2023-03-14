from pathlib import Path
import sys
import  pickle

import numpy as np

DIRNAME = Path(__file__).absolute().resolve().parent
sys.path.append(str(DIRNAME))
sys.path.append(str(DIRNAME.parent.parent.parent))


import qlib
from qlib.constant import REG_CN, REG_US
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.tests.data import GetData
from qlib.tests.config import CSI300_BENCH, CSI300_DATASET_CONFIG
from qlib.contrib.graph import stock_concept_data, stock_stock_data
from qlib.data import D
from qlib.contrib.interpreter.attentionx import AttentionX


def load_graph(market, instrument, start_date, end_date, relation_source='industry'):
    universe = D.list_instruments(
        D.instruments(instrument), start_time=start_date, end_time=end_date
    )
    stocks_sorted_list = sorted(list(universe.keys()))
    stocks_index_dict = {}
    for i, stock in enumerate(stocks_sorted_list):
        stocks_index_dict[stock] = i
    n = len(stocks_index_dict.keys())
    if relation_source == 'stock-stock':
        return stock_stock_data.get_all_matrix(
            market, stocks_index_dict,
            #data_path="D:/Code/myqlib/.qlib/qlib_data/graph_data/"
            data_path='/home/zhangzexi/.qlib/qlib_data/graph_data/' # This is the data path on the server
        ), stocks_sorted_list
    elif relation_source == 'industry':
        industry_dict = stock_concept_data.read_graph_dict(
            market,
            relation_name="SW_belongs_to",
            #data_path="D:/Code/myqlib/.qlib/qlib_data/graph_data/",
            data_path='/home/zhangzexi/.qlib/qlib_data/graph_data/'
        )
        return stock_concept_data.get_full_connection_matrix(
            industry_dict, stocks_index_dict
        ), stocks_sorted_list
    elif relation_source == 'full':
        return np.ones(shape=(n, n)), stocks_sorted_list
    else:
        raise ValueError("unknown graph name `%s`" % relation_source)


def make_config():
    train_start_date = '2008-01-01'
    train_end_date = '2014-12-31'
    valid_start_date = '2015-01-01'
    valid_end_date = '2016-12-31'
    test_start_date = '2017-01-01'
    #explain_end_date = '2017-01-03'
    test_end_date = '2020-08-01'
    #test_end_date = explain_end_date
    explain_end_date = '2017-02-01' # for each timestamp, attentionx requires about 3 sec to generate explanation.

    market = "A_share"
    instrument = "csi300"

    #rel_encoding, stock_name_list = load_graph(market, instrument, train_start_date, test_end_date, 'industry')
    rel_encoding, stock_name_list = load_graph(market, instrument, train_start_date, test_end_date, 'stock-stock')
    #rel_encoding, stock_name_list = load_graph(market, instrument, train_start_date, test_end_date, 'full')

    config = {}
    config['model'] = {
        "class": "Graphs",
        "module_path": "qlib.contrib.model.pytorch_ts_dgl",
        "kwargs": {
            "graph_model": "GAT", # or 'simpleHGN'
            "d_feat": 6,
            "hidden_size": 64,
            "num_layers": 1,
            "loss": "mse",
            "dropout": 0.7,
            "n_epochs": 100,
            "metric": "loss",
            "base_model": "LSTM",
            "use_residual": True, # Note that if use residual connection, the explanation fidelity can be near to zero.
            "GPU": 1,
            "lr": 1e-4,
            "early_stop": 10,
            "rel_encoding": rel_encoding,
            "stock_name_list": stock_name_list
        },
    }
    config['log'] = {"class": "Graphs",
        "module_path": "qlib.contrib.model.pytorch_ts_dgl",
        "kwargs": {
            "graph_model": "GAT", # or 'simpleHGN'
            "d_feat": 6,
            "hidden_size": 64,
            "num_layers": 1,
            "loss": "mse",
            "dropout": 0.7,
            "n_epochs": 100,
            "metric": "loss",
            "base_model": "LSTM",
            "use_residual": True,
            "GPU": 1,
            "lr": 1e-4,
            "early_stop": 10}}

    dh_config = {
        'start_time': train_start_date, 'end_time': test_end_date,
        'fit_start_time': train_start_date, 'fit_end_time': train_end_date,
        'infer_processors': [{'class': 'RobustZScoreNorm',
                              'kwargs': {'clip_outlier': True,
                                         'fields_group': 'feature'}},
                             {'class': 'Fillna',
                              'kwargs': {'fill_value': 0,
                                         'fields_group': 'feature'}}],
        'instruments': instrument,
        'label': ['Ref($close, -2) / Ref($close, -1) - 1'],
        'learn_processors': [{'class': 'DropnaLabel'},
                             {'class': 'CSRankNorm',
                              'kwargs': {'fields_group': 'label'}}],
    }


    handler = {
        "class": "Alpha360",
        'module_path': 'qlib.contrib.data.handler',
        'kwargs': dh_config
    }


    dataset_config = {
        'class': 'DatasetH',
        'module_path': 'qlib.data.dataset',
        'kwargs': {'handler': handler,
                   'segments': {
                       'train': (train_start_date, train_end_date),
                       'valid': (valid_start_date, valid_end_date),
                       'test': (test_start_date, test_end_date),
                       'explain':(test_start_date, explain_end_date)
                   },
                   }}
                   #'step_len':20}}
    config['dataset'] = dataset_config
    return config

def make_port_config(model, dataset):
    return {
        "executor": {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": "day",
                "generate_portfolio_metrics": True,
            },
        },
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {
                "signal": (model, dataset),
                "topk": 50,
                "n_drop": 5,
            },
        },
        "backtest": {
            "start_time": "2017-01-01",
            "end_time": "2020-08-01",
            "account": 100000000,
            "benchmark": CSI300_BENCH,
            "exchange_kwargs": {
                "freq": "day",
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
            },
        },
    }



if __name__ == "__main__":

    # use default data
    #provider_uri = "D:/Code/myqlib/.qlib/qlib_data/cn_data"  # target_dir
    provider_uri = '/home/zhangzexi/.qlib/qlib_data/cn_data/'
    GetData().qlib_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    config = make_config()
    model = init_instance_by_config(config['model'])
    dataset = init_instance_by_config(config['dataset'])

    port_analysis_config = make_port_config(model, dataset)
    # NOTE: This line is optional
    # It demonstrates that the dataset can be used standalone.
    #example_df = dataset.prepare("train")
    #print(example_df.head())


    # start exp
    with R.start(experiment_name="gats_dgl"):
        R.log_params(**flatten_dict(config['log']))
        model.fit(dataset)
        R.save_objects(**{"params.pkl": model})

        # prediction
        recorder = R.get_recorder()
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        # Signal Analysis
        sar = SigAnaRecord(recorder)
        sar.generate()

        # backtest. If users want to use backtest based on their own prediction,
        # please refer to https://qlib.readthedocs.io/en/latest/component/recorder.html#record-template.
        par = PortAnaRecord(recorder, port_analysis_config, "day")
        par.generate()

        # get attention explanation
        explainer = AttentionX(graph_model='homograph', num_layers=1, device=model.device)

        explanation, scores = model.get_explanation(dataset, explainer)
        config['log']['explainer'] = 'AttentionX'
        config['log']['explanation'] = explanation
        config['log']['scores'] = scores
        print('Saving explanations...')
        with open("explanation", 'wb') as f:  # 打开文件
            pickle.dump(config['log'], f)





