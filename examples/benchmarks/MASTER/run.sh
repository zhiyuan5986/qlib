if [ ! -d "./logs" ]; then
    mkdir ./logs
fi
if [ ! -d "./backtest" ]; then
    mkdir ./backtest
fi
universe=csi300
only_backtest=true
if $only_backtest; then
    nohup python -u main.py --universe $universe --only_backtest > ./backtest/${universe}.log 2>&1 &
else
    nohup python -u main.py --universe $universe > ./logs/${universe}.log 2>&1 &
fi
echo $!