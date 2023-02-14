# Introduction
Anonymous framework now

# Dataset
Following DDG-DA, we run experiments on the crowd source version of qlib data which can be downloaded by
```bash
wget https://github.com/chenditc/investment_data/releases/download/20220720/qlib_bin.tar.gz
tar -zxvf qlib_bin.tar.gz -C ~/.qlib/qlib_data/crowd_data --strip-components=2
```
Then, argument `--data_dir crowd_data` and `--data_dir cn_data` for crowd-source data and Yahoo-source data, respectively.

Argument `--alpha 360` or `--alpha 158` for Alpha360 and Alpha 158, respectively.
