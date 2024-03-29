if [ ! -d "./logs" ]; then
    mkdir ./logs
fi
if [ ! -d "./backtest" ]; then
    mkdir ./backtest
fi
universe=csi300
# add a line to use `sed` to change all the 'csi300' in `filename.yaml` into 'csi500'
# sed -i 's/csi300/csi500/g' filename.yaml
# add a line to use `sed` to change all the 'csi300' or 'csi500' in `filename.yaml` into `universe`
sed -i "s/csi300\|csi500/$universe/g" workflow_config_master_Alpha158.yaml
only_backtest=false
if $only_backtest; then
    nohup python -u main.py --universe $universe --only_backtest > ./backtest/${universe}.log 2>&1 &
else
    nohup python -u main.py --universe $universe > ./logs/${universe}.log 2>&1 &
fi
echo $!