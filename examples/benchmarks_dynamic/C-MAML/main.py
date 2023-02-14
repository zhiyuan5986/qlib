# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import copy
from pathlib import Path
import torch

from qlib.data.dataset import Dataset, TSDataSampler

from qlib.workflow.task.gen import RollingGen
from qlib.utils import init_instance_by_config
from qlib.data.dataset.handler import DataHandlerLP

from src.model import MetaModelDS
import fire
import sys
DIRNAME = Path(__file__).absolute().resolve().parent
sys.path.append(str(DIRNAME))
sys.path.append(str(DIRNAME.parent.parent.parent))
from examples.benchmarks_dynamic.baseline.benchmark import Benchmark
from examples.benchmarks_dynamic.incremental.main import Incremental, preprocess
from examples.benchmarks_dynamic.incremental.src.dataset import MetaDatasetDS


class OKASA(Incremental):

    def __init__(self, data_dir='cn_data', market='csi300', horizon=1, alpha=360, step=20, rank_label=True,
                 forecast_model="linear", tag='', first_order=True):
        super().__init__(data_dir=data_dir, market=market, horizon=horizon, alpha=alpha,
                       step=step, rank_label=rank_label, forecast_model=forecast_model, tag=tag, first_order=first_order)

    @property
    def meta_exp_name(self):
        return f"CMAML_{self.market}_{self.forecast_model}_alpha{self.alpha}_horizon{self.horizon}_step{self.step}_rank{self.rank_label}_{self.tag}"


    def dump_data(self):
        segments = self.task["dataset"]["kwargs"]["segments"]

        t = copy.deepcopy(self.task)
        t['dataset']["kwargs"]["segments"]['train'] = (segments["train"][0], segments['test'][1])
        ds = init_instance_by_config(t['dataset'], accept_types=Dataset)
        data = ds.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        if t['dataset']['class'] == 'TSDatasetH':
            data.config(fillna_type="ffill+bfill")  # process nan brought by dataloader
            # self.L = rolling_task['dataset']['kwargs']['step_len']
        # else:
        #     data = None

        rolling_task = self.rb.basic_task()
        if 'pt_model_kwargs' in rolling_task['model']['kwargs'] and rolling_task['model']['class'] != 'DNNModelPytorch':
            self.d_feat = rolling_task['model']['kwargs']['pt_model_kwargs']['input_dim']
        elif 'd_feat' in rolling_task['model']['kwargs']:
            self.d_feat = rolling_task['model']['kwargs']['d_feat']
        else:
            self.d_feat = 6 if self.alpha == 360 else 20

        trunc_days = self.horizon if self.data_dir == 'us_data' else (self.horizon + 1)
        gen = RollingGen(step=self.step, rtype=RollingGen.ROLL_SD)
        segments = rolling_task["dataset"]["kwargs"]["segments"]
        train_begin = segments["train"][0]
        train_end = gen.ta.get(gen.ta.align_idx(train_begin) + gen.step - 1)
        test_begin = gen.ta.get(gen.ta.align_idx(train_begin) + gen.step - 1 + trunc_days)
        test_end = segments["valid"][1]
        test_end = gen.ta.get(gen.ta.align_idx(test_end) - gen.step)
        # extra_begin = gen.ta.get(gen.ta.align_idx(train_end) + 1)
        # extra_end = gen.ta.get(gen.ta.align_idx(test_begin) - 1)
        seperate_point = str(rolling_task["dataset"]["kwargs"]["segments"]["valid"][0])
        rolling_task["dataset"]["kwargs"]["segments"] = {
            "train": (segments["train"][0], train_end),
            # "extra": (extra_begin, extra_end),
            "test": (test_begin, test_end),
        }

        kwargs = dict(
            task_tpl=rolling_task,
            step=self.step,
            # segments=0.7,  # keep test period consistent with the dataset yaml
            segments=seperate_point,
            # trunc_days=self.horizon+1,
            task_mode='train',
        )
        if self.forecast_model == 'MLP' and self.alpha == 158:
            kwargs.update(task_mode='test')
        md = MetaDatasetDS(data=data, **kwargs)
        md.meta_task_l = preprocess(md.meta_task_l, d_feat=self.d_feat,
                                     is_mlp=self.forecast_model == 'MLP', alpha=self.alpha,
                                     step=self.step, H=self.horizon if self.data_dir == 'us_data' else (1+self.horizon),
                                    need_permute=not self.forecast_model in ['TCN'])
        phases = ["train", "test"]
        meta_tasks_train, meta_tasks_valid = md.prepare_tasks(phases)
        self.L = meta_tasks_train[0].get_meta_input()['X_test'].shape[1]

        test_begin = segments["valid"][0]
        # train_end = gen.ta.get(gen.ta.align_idx(train_begin) + gen.step - 1)
        # test_begin = gen.ta.get(gen.ta.align_idx(train_begin) + gen.step - 1 + trunc_days)
        rolling_task["dataset"]["kwargs"]["segments"] = {
            "test": (test_begin, segments['test'][1]),
            # "extra": (extra_begin, extra_end),
            # "test": (test_begin, segments['test'][1]),
        }
        if trunc_days > 1:
            extra_end = gen.ta.get(gen.ta.align_idx(test_begin) - 1)
            extra_begin = gen.ta.get(gen.ta.align_idx(test_begin) - trunc_days + 1)
            rolling_task["dataset"]["kwargs"]["segments"]["train"] = (extra_begin, extra_end)

        kwargs.update(task_tpl=rolling_task, segments=0.0)
        if self.forecast_model == 'MLP' and self.alpha == 158:
            kwargs.update(task_mode='test')
            data_I = ds.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        else:
            data_I = None
        md2 = MetaDatasetDS(data=data, data_I=data_I, **kwargs)
        md2.meta_task_l = preprocess(md2.meta_task_l, d_feat=self.d_feat,
                                     is_mlp=self.forecast_model == 'MLP', alpha=self.alpha,
                                     step=self.step, H=self.horizon if self.data_dir == 'us_data' else (1+self.horizon),
                                    need_permute=not self.forecast_model in ['TCN'])
        meta_tasks_test = md2.prepare_tasks('test')
        return meta_tasks_train, meta_tasks_valid, meta_tasks_test


    def offline_training(self, seed=43):
        """
        training a src model based on a simplified linear proxy model;
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        if self.forecast_model in ['TCN', ] and self.alpha == 360:
            model = Benchmark(data_dir=self.data_dir, model_type=self.forecast_model, alpha=self.alpha,
                                  market=self.market, rank_label=self.rank_label).get_fitted_model(f"_{seed}")
        else:
            model = None

        # with R.start(experiment_name=self.meta_exp_name):
        mm = MetaModelDS(self.task, sample_num=8000 if self.market == 'csi500' else 5000,
                         is_seq=self.is_rnn, d_feat=self.d_feat,
                         alpha=self.alpha, lr=0.01,
                         first_order=self.first_order, num_head=self.num_head, temperature=self.temperature,
                         pretrained_model=model)
        mm.fit(self.meta_tasks_train, self.meta_tasks_valid)
            # R.save_objects(model=mm)
        return mm

if __name__ == "__main__":
    fire.Fire(OKASA)
