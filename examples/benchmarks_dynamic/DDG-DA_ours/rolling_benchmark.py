# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import time
from pprint import pprint
from typing import Callable, List

import numpy as np
import pandas as pd
import torch

from pathlib import Path
import sys
DIRNAME = Path(__file__).absolute().resolve().parent
sys.path.append(str(DIRNAME.parent.parent.parent))

import qlib

from qlib.config import C

from qlib.data.dataset.weight import Reweighter

from qlib.data.dataset import Dataset, DataHandlerLP, TSDataSampler

from qlib.model import Model

from qlib.model.ens.ensemble import RollingEnsemble
from qlib.utils import init_instance_by_config, auto_filter_kwargs, fill_placeholder
import fire
import yaml
from qlib import auto_init, get_module_logger
from pathlib import Path
from tqdm.auto import tqdm
from qlib.model.trainer import TrainerR, _log_task_info
from qlib.workflow import R, Recorder, Experiment
from qlib.tests.data import GetData

DIRNAME = Path(__file__).absolute().resolve().parent
from qlib.workflow.task.gen import task_generator, RollingGen
from qlib.workflow.task.collect import RecorderCollector
from qlib.workflow.record_temp import PortAnaRecord, SigAnaRecord


class RollingBenchmark:
    """
    **NOTE**
    before running the example, please clean your previous results with following command
    - `rm -r mlruns`

    """

    def __init__(
        self,
        data_dir="cn_data",
        market="csi300",
        init_data=True,
        model_type="linear",
        step=20,
        alpha="158",
        horizon=1,
        rank_label=True,
    ) -> None:
        self.data_dir = data_dir
        self.market = market
        if init_data:
            if data_dir == "cn_data":
                GetData().qlib_data(target_dir="~/.qlib/qlib_data/cn_data", exists_skip=True)
                auto_init()
            else:
                qlib.init(
                    provider_uri="~/.qlib/qlib_data/" + data_dir, region="us" if self.data_dir == "us_data" else "cn",
                )
        self.step = step
        self.horizon = horizon
        # self.rolling_exp = rolling_exp
        self.model_type = model_type
        self.rank_label = rank_label
        self.alpha = alpha
        self.tag = ""
        if self.data_dir == "us_data":
            self.benchmark = "^gspc"
        elif self.market == "csi500":
            self.benchmark = "SH000905"
        elif self.market == "csi100":
            self.benchmark = "SH000903"
        else:
            self.benchmark = "SH000300"

    @property
    def rolling_exp(self):
        return f"rolling_{self.data_dir}_{self.market}_{self.model_type}_alpha{self.alpha}_h{self.horizon}_step{self.step}_rank{self.rank_label}_{self.tag}"

    @property
    def COMB_EXP(self):
        return "final_" + self.rolling_exp

    def basic_task(self):
        """For fast training rolling"""
        if self.model_type == "gbdt":
            conf_path = (
                DIRNAME.parent.parent
                / "benchmarks"
                / "LightGBM"
                / "workflow_config_lightgbm_Alpha{}.yaml".format(self.alpha)
            )
            # dump the processed data on to disk for later loading to speed up the processing
            filename = "lightgbm_alpha{}_handler_horizon{}.pkl".format(self.alpha, self.horizon)
        elif self.model_type == "linear":
            conf_path = (
                DIRNAME.parent.parent
                / "benchmarks"
                / "Linear"
                / "workflow_config_linear_Alpha{}.yaml".format(self.alpha)
            )
            # dump the processed data on to disk for later loading to speed up the processing
            filename = "linear_alpha{}_handler_horizon{}.pkl".format(self.alpha, self.horizon)
        elif self.model_type == "mlp":
            conf_path = (
                DIRNAME.parent.parent / "benchmarks" / "MLP" / "workflow_config_mlp_Alpha{}.yaml".format(self.alpha)
            )
            # dump the processed data on to disk for later loading to speed up the processing
            filename = "mlp_alpha{}_handler_horizon{}.pkl".format(self.alpha, self.horizon)
        else:
            conf_path = (
                DIRNAME.parent.parent
                / "benchmarks"
                / self.model_type
                / "workflow_config_{}_Alpha{}.yaml".format(self.model_type.lower(), self.alpha)
            )
            filename = "alpha{}_handler_horizon{}.pkl".format(self.alpha, self.horizon)
            # raise AssertionError("Model type is not supported!")

        filename = f"{self.data_dir}_{self.market}_rank{self.rank_label}_{filename}"
        h_path = DIRNAME.parent / "baseline" / filename

        with conf_path.open("r") as f:
            conf = yaml.safe_load(f)

        # modify dataset horizon
        conf["task"]["dataset"]["kwargs"]["handler"]["kwargs"]["label"] = [
            "Ref($close, -{}) / Ref($close, -1) - 1".format(self.horizon + 1)
        ]

        if self.market != "csi300":
            conf["task"]["dataset"]["kwargs"]["handler"]["kwargs"]["instruments"] = self.market
            if self.data_dir == "us_data":
                conf["task"]["dataset"]["kwargs"]["handler"]["kwargs"]["label"] = [
                    "Ref($close, -{}) / $close - 1".format(self.horizon)
                ]

        batch_size = 5000
        if self.market == "csi100":
            batch_size = 2000
        elif self.market == "csi500":
            batch_size = 8000

        for k, v in {"early_stop": 8, "batch_size": batch_size, "lr": 0.001, "seed": None,}.items():
            if k in conf["task"]["model"]["kwargs"]:
                conf["task"]["model"]["kwargs"][k] = v
        if conf["task"]["model"]["class"] == "TransformerModel":
            conf["task"]["model"]["kwargs"]["dim_feedforward"] = 32
            conf["task"]["model"]["kwargs"]["reg"] = 0

        task = conf["task"]

        if not h_path.exists():
            h_conf = task["dataset"]["kwargs"]["handler"]
            if not self.rank_label and not (self.model_type == "gbdt" or self.alpha == 158):
                proc = h_conf["kwargs"]["learn_processors"][-1]
                if (
                    isinstance(proc, str)
                    and proc == "CSRankNorm"
                    or isinstance(proc, dict)
                    and proc["class"] == "CSRankNorm"
                ):
                    h_conf["kwargs"]["learn_processors"] = h_conf["kwargs"]["learn_processors"][:-1]
                    print("Remove CSRankNorm")
                    h_conf["kwargs"]["learn_processors"].append(
                        {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}}
                    )

            print(h_conf)
            h = init_instance_by_config(h_conf)
            h.to_pickle(h_path, dump_all=True)

        task["dataset"]["kwargs"]["handler"] = f"file://{h_path}"
        task["record"] = ["qlib.workflow.record_temp.SignalRecord"]
        return task

    def create_rolling_tasks(self):
        task = self.basic_task()
        task_l = task_generator(
            task, RollingGen(step=self.step, trunc_days=self.horizon + 1)
        )  # the last two days should be truncated to avoid information leakage
        return task_l

    def train_rolling_tasks(self, task_l=None):
        if task_l is None:
            task_l = self.create_rolling_tasks()
        trainer = TrainerR(experiment_name=self.rolling_exp)
        trainer(task_l)

    def ens_rolling(self):
        rc = RecorderCollector(
            experiment=self.rolling_exp,
            artifacts_key=["pred", "label"],
            process_list=[RollingEnsemble()],
            # rec_key_func=lambda rec: (self.COMB_EXP, rec.info["id"]),
            artifacts_path={"pred": "pred.pkl", "label": "label.pkl"},
        )
        res = rc()

        with R.start(experiment_name=self.COMB_EXP):
            R.log_params(exp_name=self.rolling_exp)
            R.save_objects(**{"pred.pkl": res["pred"], "label.pkl": res["label"]})
        return res["pred"]

    def update_rolling_rec(self, preds=None):
        """
        Evaluate the combined rolling results
        """
        backtest_config = {
            "strategy": {
                "class": "TopkDropoutStrategy",
                "module_path": "qlib.contrib.strategy",
                "kwargs": {"signal": "<PRED>", "topk": 50, "n_drop": 5},
            },
            "backtest": {
                "start_time": None,
                "end_time": None,
                "account": 100000000,
                "benchmark": self.benchmark,
                "exchange_kwargs": {
                    "limit_threshold": None if self.data_dir == "us_data" else 0.095,
                    "deal_price": "close",
                    "open_cost": 0.0005,
                    "close_cost": 0.0015,
                    "min_cost": 5,
                },
            },
        }
        rec = R.get_exp(experiment_name=self.COMB_EXP).list_recorders(rtype=Experiment.RT_L)[0]
        SigAnaRecord(recorder=rec, skip_existing=True).generate()
        PortAnaRecord(recorder=rec, config=backtest_config, skip_existing=True).generate()
        label = init_instance_by_config(self.basic_task()["dataset"], accept_types=Dataset).\
            prepare(segments="test", col_set="label", data_key=DataHandlerLP.DK_L)
        if isinstance(label, TSDataSampler):
            label = pd.DataFrame({'label': label.data_arr[:-1][:, 0]}, index=label.data_index)
        else:
            label.columns = ['label']
        label['pred'] = preds.loc[label.index]
        mse = ((label['pred'].to_numpy() - label['label'].to_numpy()) ** 2).mean()
        mae = np.abs(label['pred'].to_numpy() - label['label'].to_numpy()).mean()
        rec.log_metrics(mse=mse, mae=mae)
        print(f"Your evaluation results can be found in the experiment named `{self.COMB_EXP}`.")
        return rec

    def run_all(self, task_l=None):
        # the results will be  save in mlruns.
        # 1) each rolling task is saved in rolling_models
        self.train_rolling_tasks(task_l)
        # 2) combined rolling tasks and evaluation results are saved in rolling
        preds = self.ens_rolling()
        rec = self.update_rolling_rec(preds)
        return rec

    def run_exp(self):
        all_metrics = {
            k: []
            for k in [
                'mse', 'mae',
                "IC",
                "ICIR",
                "Rank IC",
                "Rank ICIR",
                "1day.excess_return_with_cost.annualized_return",
                "1day.excess_return_with_cost.information_ratio",
                "1day.excess_return_with_cost.max_drawdown",
            ]
        }
        test_time = []
        for i in range(10):
            np.random.seed(i + 43)
            torch.manual_seed(i + 43)
            torch.cuda.manual_seed(i + 43)
            start_time = time.time()
            self.tag = str(time.time())
            rec = self.run_all()
            test_time.append(time.time() - start_time)
            # exp = R.get_exp(experiment_name=self.COMB_EXP)
            # rec = exp.list_recorders(rtype=exp.RT_L)[0]
            metrics = rec.list_metrics()
            for k in all_metrics.keys():
                all_metrics[k].append(metrics[k])
            pprint(all_metrics)

        with R.start(
            experiment_name=f"final_{self.data_dir}_{self.market}_{self.alpha}_{self.horizon}_{self.model_type}"
        ):
            R.save_objects(all_metrics=all_metrics)
            test_time = np.array(test_time)
            R.log_metrics(test_time=test_time)
            print(f"Time cost: {test_time.mean()}")
            res = {}
            for k in all_metrics.keys():
                v = np.array(all_metrics[k])
                res[k] = [v.mean(), v.std()]
                R.log_metrics(**{"final_" + k: res[k]})
            pprint(res)
        test_time = np.array(test_time)
        print(f"Time cost: {test_time.mean()}")


if __name__ == "__main__":
    # GetData().qlib_data(exists_skip=True)
    # auto_init()
    fire.Fire(RollingBenchmark)
