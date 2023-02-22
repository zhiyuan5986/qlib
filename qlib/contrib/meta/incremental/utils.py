import warnings
from collections import defaultdict
import numpy as np
import torch
from typing import List

from qlib.data.dataset import TSDataSampler


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
        warnings.warn(f"{e} when processing segment {seg}")
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
    X_test = meta_input["X_test"]
    y_test = meta_input["y_test"]
    mask_x = torch.isnan(X_test).sum(-1) == 0
    meta_input["X_test"] = X_test[mask_x]

    y_test = y_test[mask_x]

    mask_y = ~torch.isnan(y_test)
    meta_input["mask_y"] = mask_y
    meta_input["y_test"] = y_test[mask_y]
    test_idx = meta_input["test_idx"]
    meta_input["test_idx"] = test_idx[np.arange(len(test_idx))[mask_x][mask_y]]
    return meta_input


def preprocess(
    task_list, d_feat=6, is_mlp=False, alpha=360, step=20, H=1, need_permute=True, need_flatten=False,
):
    skip = []
    for i, task in enumerate(task_list):
        meta_input = task.get_meta_input()
        data_type = set()
        for k in meta_input.keys():
            if k.startswith("X") or k.startswith("y"):
                data_type.add(k[2:])
                if isinstance(meta_input[k], np.ndarray):
                    meta_input[k] = torch.tensor(meta_input[k], dtype=torch.float32)
                else:
                    meta_input[k] = torch.tensor(meta_input[k].to_numpy(), dtype=torch.float32)
        # if task.processed_meta_input['y_test'].shape[0] == 0:
        #     skip.append(i)
        if is_mlp and alpha == 158:
            _mask_mlp158(meta_input)

        if need_flatten:
            if alpha == 158:
                for dt in data_type:
                    k = "X_" + dt
                    meta_input[k] = meta_input[k].reshape(len(meta_input[k]), -1)
        elif alpha == 360:
            for dt in data_type:
                k = "X_" + dt
                meta_input[k] = meta_input[k].reshape(len(meta_input[k]), d_feat, -1)
                if need_permute:
                    meta_input[k] = meta_input[k].permute(0, 2, 1)

        test_date = meta_input["test_idx"].codes[0] - meta_input["test_idx"].codes[0][0]
        meta_input["meta_end"] = (test_date <= (test_date[-1] - H + 1)).sum()
        # _additional_info(meta_input, 'train')
        # _additional_info(meta_input, 'test')
        # match = stock_train_ids.unsqueeze(-1) == stock_train_ids.unsqueeze(0)

        # meta_input['mu'] = meta_input['X'].mean(0, keepdim=True)
        # meta_input['std'] = meta_input['X'].std(0, keepdim=True)
        # meta_input['train_mu'] = meta_input['X_train'].mean(1, keepdim=True)
        # meta_input['test_mu'] = meta_input['X_test'].mean(1, keepdim=True)
        # meta_input['train_std'] = meta_input['X_train'].std(1, keepdim=True)
        # meta_input['test_std'] = meta_input['X_test'].std(1, keepdim=True)

        # meta_input['X_train'] = (meta_input['X_train'] - meta_input['train_mu']) / (meta_input['train_std'] + 1e-9)
        # meta_input['X_test'] = (meta_input['X_test'] - meta_input['test_mu']) / (meta_input['test_std'] + 1e-9)

    # if skip:
    #     for i in skip:
    #         for k in ['X', 'y']:
    #             task_list[i+1].processed_meta_input[k] = task_list[i].processed_meta_input[k]
    #     task_list = [task_list[i] for i in range(len(task_list)) if i not in skip]
    return task_list


def _additional_info(meta_input: dict, key_prefix="train"):
    date_ids, stock_ids = meta_input[key_prefix + "_idx"].codes
    mu, std = [], []
    # pos_ids, neg_ids = [], []
    for date_id in np.arange(date_ids[0], date_ids[-1] + 1):
        mask = [date_ids == date_id]
        X = meta_input["X_train" if key_prefix == "train" else "X_test"][mask]
        # mu.append(X.mean(0, keepdim=True).repeat([len(X)] + [1] * (X.dim()-1)))
        # std.append(X.std(0, keepdim=True).repeat([len(X)] + [1] * (X.dim()-1)))
        mu.append(X.mean(1, keepdim=True).mean(0, keepdim=True).repeat([len(X)] + [1] * (X.dim() - 1)))
        std.append(X.std(1, keepdim=True).mean(0, keepdim=True).repeat([len(X)] + [1] * (X.dim() - 1)))
    mu = torch.cat(mu, 0)
    std = torch.cat(std, 0)

    meta_input.update(
        {
            # key_prefix + '_date_id': torch.LongTensor(date_ids - date_ids.min()),
            key_prefix + "_mu": mu,
            key_prefix + "_std": std,
        }
    )
    return stock_ids
