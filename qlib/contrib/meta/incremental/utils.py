import traceback
import warnings
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from typing import List
import bisect

from torch import nn

from qlib.data.dataset import TSDataSampler

def get_data(data: TSDataSampler, segment: List):
    # ext_slice = TSDatasetH._extend_slice(slice(segment), )
    # 我希望的data是向前追溯了step_len的，start是segment[0], end是segment[1]
    cal = data.data_index.get_level_values('datetime').unique()
    start_date = cal[max(bisect.bisect_left(cal,pd.Timestamp(segment[0]))-data.step_len, 0)]

    d_idx_with_step = data.idx_df.loc(axis=0)[start_date: segment[1]].to_numpy().flatten()
    d_idx = data.idx_df.loc(axis=0)[segment[0]: segment[1]].to_numpy().flatten()
    if len(d_idx) == 0:
        return None, None
    d_idx_with_step = d_idx_with_step[~np.isnan(d_idx_with_step.astype(np.float64))]
    d_idx = d_idx[~np.isnan(d_idx.astype(np.float64))]
    # use the code in __getitem__ of TSDataSampler to get the raw np.ndarray data from data.data_arr

    # indices = [range(data.idx_df.loc(segment[0]), int(d_idx[-1])+1)]
    # indices = [range(int(data._get_indices(*data._get_row_col(d_idx[0]))[0]), int(d_idx[-1])+1)]
    # indices = [data._get_indices(*data._get_row_col(d_idx[0])), d_idx[1:]]
    # indices = [data._get_indices(*data._get_row_col(i)) for i in d_idx]
    # indices = np.concatenate(indices)
    d_idx_with_step = np.nan_to_num(d_idx_with_step.astype(np.float64), nan=data.nan_idx).astype(int)
    df = pd.DataFrame(data.data_arr[d_idx_with_step], index=data.data_index[d_idx_with_step.astype(int)])
    tsds = TSDataSampler(
        data = df,
        start = segment[0],
        end = segment[1],
        step_len = data.step_len,
        fillna_type="ffill+bfill"
    )
    return tsds, tsds.get_index()

    if len(d_idx) == 0:
        return None, None
    d_idx = d_idx[~np.isnan(d_idx.astype(np.float64))]
    d = data[d_idx]
    indices = data.data_index[d_idx.astype(int)]

    start_date = data.data_index[data.data_index.get_loc(segment[0])-data.step_len]
    end_date = data.data_index.get_loc(segment[1])
    d_idx = data.idx_df.loc(axis=0)[start_date : segment[0]].to_numpy().flatten()
    return TSDataSampler(
        data = data
    )
    d = data.loc(axis=0)[segment[0] : segment[1]]
    return d

# def get_data(data: TSDataSampler, segment: List):
#     # ext_slice = TSDatasetH._extend_slice(slice(segment), )
#     # 我希望的data是向前追溯了step_len的，start是segment[0], end是segment[1]
#     cal = data.data_index.get_level_values('datetime').unique()

#     d_idx = data.idx_df.loc(axis=0)[segment[0] : segment[1]].to_numpy().flatten()
#     if len(d_idx) == 0:
#         return None, None
#     d_idx = d_idx[~np.isnan(d_idx.astype(np.float64))]
#     # use the code in __getitem__ of TSDataSampler to get the raw np.ndarray data from data.data_arr

#     indices = [range(data.idx_df.loc(segment[0]), int(d_idx[-1])+1)]
#     # indices = [range(int(data._get_indices(*data._get_row_col(d_idx[0]))[0]), int(d_idx[-1])+1)]
#     # indices = [data._get_indices(*data._get_row_col(d_idx[0])), d_idx[1:]]
#     # indices = [data._get_indices(*data._get_row_col(i)) for i in d_idx]
#     indices = np.concatenate(indices)
#     indices = np.nan_to_num(indices.astype(np.float64), nan=data.nan_idx).astype(int)
#     df = pd.DataFrame(data.data_arr[indices], index=data.data_index[d_idx.astype(int)])
#     return TSDataSampler(
#         data = df,
#         start = segment[0],
#         end = segment[1],
#         step_len = data.step_len,
#         fillna_type="ffill+bfill"
#     ), df.index

#     if len(d_idx) == 0:
#         return None, None
#     d_idx = d_idx[~np.isnan(d_idx.astype(np.float64))]
#     d = data[d_idx]
#     indices = data.data_index[d_idx.astype(int)]

#     start_date = data.data_index[data.data_index.get_loc(segment[0])-data.step_len]
#     end_date = data.data_index.get_loc(segment[1])
#     d_idx = data.idx_df.loc(axis=0)[start_date : segment[0]].to_numpy().flatten()
#     return TSDataSampler(
#         data = data
#     )
#     d = data.loc(axis=0)[segment[0] : segment[1]]
#     return d

def get_data_and_idx(data: TSDataSampler, segment: List):
    d_idx = data.idx_df.loc(axis=0)[segment[0] : segment[1]].to_numpy().flatten()
    if len(d_idx) == 0:
        return None, None
    d_idx = d_idx[~np.isnan(d_idx.astype(np.float64))]
    d = data[d_idx]
    indices = data.data_index[d_idx.astype(int)]
    return d, indices


def get_data_from_seg(seg, data, test=False):
    try:
        d = (
            data.loc(axis=0)[seg[0] : seg[1]]
            if not test or seg[1] <= str(data.index[-1][0])
            else data.loc(axis=0)[seg[0] :]
        )
    except Exception as e:
        traceback.print_exc()
        new_seg = [seg[0], seg[1]]
        all_dates = data.index.levels[0]
        if seg[0] not in all_dates:
            new_seg[0] = all_dates[all_dates > seg[0]][0]
            if str(new_seg[0])[:10] > seg[1]:
                warnings.warn(f"Exceed test time{new_seg}")
                return None
        if seg[1] not in all_dates:
            new_seg[1] = all_dates[all_dates < seg[1]][-1]
            if str(new_seg[1])[:10] < seg[0]:
                warnings.warn(f"Exceed training time{new_seg}")
                return None
            d = (
                data.loc(axis=0)[new_seg[0] : new_seg[1]]
                if not test or new_seg[1] <= all_dates[-1]
                else data.loc(axis=0)[new_seg[0] :]
            )
        else:
            d = (
                data.loc(axis=0)[new_seg[0] : new_seg[1]]
                if not test or new_seg[1] <= str(all_dates[-1])
                else data.loc(axis=0)[new_seg[0] :]
            )
        warnings.warn(f"{seg} becomes {new_seg} after adjustment")
    return d


def override_state(groups, new_opt):
    saved_groups = new_opt.param_groups
    id_map = {old_id: p for old_id, p in zip(range(len(saved_groups[0]["params"])), groups[0]["params"])}
    state = defaultdict(dict)
    for k, v in new_opt.state[0].items():
        if k in id_map:
            param = id_map[k]
            for _k, _v in v.items():
                state[param][_k] = _v.detach() if isinstance(_v, torch.Tensor) else _v
        else:
            state[k] = v
    return state


def _mask_mlp158(meta_input):
    # X_test = meta_input["X_test"]
    y_test = meta_input["y_test"]
    # mask_x = torch.isnan(X_test).sum(-1) == 0
    # meta_input["X_test"] = X_test[mask_x]
    # y_test = y_test[mask_x]

    mask_y = ~torch.isnan(y_test)
    meta_input["mask_y"] = mask_y
    # meta_input["y_test"] = y_test[mask_y]
    # test_idx = meta_input["test_idx"]
    # meta_input["test_idx"] = test_idx[np.arange(len(test_idx))[mask_x][mask_y]]
    return meta_input


def preprocess(
    task_list, factor_num=6, is_mlp=False, alpha=360, step=20, H=1, not_sequence=False, to_tensor=True,
):
    """
    Used to preprocess the task list
    """
    skip = []
    for i, task in enumerate(task_list):
        meta_input = task.get_meta_input()
        data_type = set()
        for k in meta_input.keys():
            if k.startswith("X") or k.startswith("y"):
                data_type.add(k[2:])
                if not isinstance(meta_input[k], np.ndarray):
                    # pd.DataFrame to numpy.ndarray
                    meta_input[k] = meta_input[k].to_numpy()
                if to_tensor:
                    # pd.DataFrame to torch.tensor
                    meta_input[k] = torch.tensor(meta_input[k], dtype=torch.float32)
        if isinstance(task.processed_meta_input["d_test"], TSDataSampler):
            if len(task.processed_meta_input['d_test']) == 0:
                skip.append(i)
        else:
            if len(task.processed_meta_input['y_test']).shape[0] == 0:
                skip.append(i)
        if is_mlp and alpha == 158:
            _mask_mlp158(meta_input)

        if not_sequence:
            # not_sequence means the input is not sequence data
            if alpha == 158:
                for dt in data_type:
                    k = "X_" + dt
                    meta_input[k] = meta_input[k].reshape(len(meta_input[k]), -1)
        elif alpha == 360:
            for dt in data_type:
                k = "X_" + dt
                meta_input[k] = meta_input[k].reshape(len(meta_input[k]), factor_num, -1)
                if isinstance(meta_input[k], torch.Tensor):
                    meta_input[k] = meta_input[k].permute(0, 2, 1)
                else:
                    meta_input[k] = meta_input[k].transpose(0, 2, 1)

        test_date = meta_input["test_idx"].codes[0] - meta_input["test_idx"].codes[0][0]
        meta_input["meta_end"] = (test_date <= (test_date[-1] - H + 1)).sum()
    if skip:
        ''' Delete tasks with empty test data '''
        j = 0
        for idx, i in enumerate(skip):
            if i < j:
                continue
            j = i + 1
            k = idx + 1
            while j == skip[k]: # loop to find continuous empty test data.
                k += 1
                j = skip[k] # should be `j += 1` ?
            for key in ['X_train', 'y_train']:
                task_list[j].processed_meta_input[key] = task_list[i].processed_meta_input[key]
        task_list = [task_list[i] for i in range(len(task_list)) if i not in skip]
    return task_list


def has_rnn(module: nn.Module):
    for module in module.modules():
        if isinstance(module, nn.RNNBase):
            return True
    return False

