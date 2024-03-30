## overview
This is an alternative version of the MASTER benchmark. 

paper: [MASTER: Market-Guided Stock Transformer for Stock Price Forecasting](https://arxiv.org/abs/2312.15235) 

codes: [https://github.com/SJTU-Quant/MASTER](https://github.com/SJTU-Quant/MASTER)

## run
You can directly use the bash script to run the codes (you can set the `universe` and `only_backtest` flag in `run.sh`), this `main.py` will test the model with 10 random seeds:
```
bash run.sh
```
<!-- or you can just directly use `qrun` tp run the codes (note that you should modify your `qlib`, since we add or modify some files in `qlib/contrib/data/dataset.py`, `qlib/data/dataset/__init__.py`, `qlib/data/dataset/processor.py` and `qlib/contrib/model/pytorch_master.py`):
```
qrun workflow_config_master_Alpha158.yaml
``` -->