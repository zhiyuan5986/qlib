# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import time
from pprint import pprint

import numpy as np
import qlib
import torch

from qlib.model.ens.ensemble import RollingEnsemble
from qlib.utils import init_instance_by_config
import fire
import yaml
from qlib import auto_init, init
from pathlib import Path
from tqdm.auto import tqdm
from qlib.model.trainer import TrainerR
from qlib.workflow import R
from qlib.tests.data import GetData

DIRNAME = Path(__file__).absolute().resolve().parent
from qlib.workflow.task.gen import task_generator, RollingGen
from qlib.workflow.task.collect import RecorderCollector
from qlib.workflow.record_temp import PortAnaRecord, SigAnaRecord, SignalRecord


class Benchmark:
    def __init__(self, data_dir="cn_data", market="csi300", model_type="linear", alpha="158", rank_label=True,) -> None:
        self.data_dir = data_dir
        self.market = market
        self.step = 20
        self.horizon = 1
        self.model_type = model_type
        self.alpha = alpha
        self.exp_name = f"{model_type}_{self.data_dir}_{self.market}_{self.alpha}_rank{rank_label}"
        self.rank_label = rank_label

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

        # if not self.rank_label:
        #     task['model']['kwargs']['loss'] = 'ic'
        task["dataset"]["kwargs"]["handler"] = f"file://{h_path}"
        return task

    def get_fitted_model(self, suffix=""):
        task = self.basic_task()
        try:
            R.set_uri("../baseline/mlruns")
            rec = list(R.list_recorders(experiment_name=self.exp_name + suffix).values())[0]
            model = rec.load_object("params.pkl")
            R.set_uri("./mlruns")
            print("Load pretrained model.")
        except:
            model = None
            model = init_instance_by_config(task["model"])
            dataset = init_instance_by_config(task["dataset"])
            # start exp
            with R.start(experiment_name=self.exp_name + suffix):
                model.fit(dataset)
                R.save_objects(**{"params.pkl": model})
        return model

    def run_all(self):
        if self.data_dir == "cn_data":
            GetData().qlib_data(target_dir="~/.qlib/qlib_data/cn_data", exists_skip=True)
            auto_init()
        else:
            init(provider_uri="~/.qlib/qlib_data/" + self.data_dir)
        task = self.basic_task()
        # model = init_instance_by_config(task["model"])
        dataset = init_instance_by_config(task["dataset"])
        # start exp
        model = self.get_fitted_model(f"_{self.seed}")
        with R.start(experiment_name=self.exp_name):
            # model.fit(dataset)
            # prediction
            recorder = R.get_recorder()
            sr = SignalRecord(model, dataset, recorder)
            sr.generate()

            # Signal Analysis
            sar = SigAnaRecord(recorder)
            sar.generate()

            # backtest. If users want to use backtest based on their own prediction,
            # please refer to https://qlib.readthedocs.io/en/latest/component/recorder.html#record-template.
            par = PortAnaRecord(recorder)
            par.generate()
        return recorder

    def run_exp(self):
        all_metrics = {
            k: []
            for k in [
                # 'rmse', 'mae',
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
            self.seed = i + 43
            rec = self.run_all()
            test_time.append(time.time() - start_time)
            # exp = R.get_exp(experiment_name=self.exp_name)
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
    fire.Fire(Benchmark)
