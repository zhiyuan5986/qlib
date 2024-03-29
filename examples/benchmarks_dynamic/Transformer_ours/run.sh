if [ ! -d "./logs" ]; then
    mkdir ./logs
fi
if [ ! -d "./backtest" ]; then
    mkdir ./backtest
fi

# set the config
universe=csi500
online_lr=0.000008
only_backtest=false

sed -i "s/csi.../$universe/g" workflow_config_transformer_Alpha158.yaml
if [ $universe == 'csi300' ]; then
    sed -i "s/SH....../SH000300/g" workflow_config_transformer_Alpha158.yaml
elif [ $universe == 'csi500' ]; then
    sed -i "s/SH....../SH000905/g" workflow_config_transformer_Alpha158.yaml
fi
if $only_backtest; then
    nohup python -u main.py --universe $universe --online_lr $online_lr --only_backtest > ./backtest/${universe}online_lr${online_lr}.log 2>&1 &
else
    nohup python -u main.py --universe $universe --online_lr $online_lr > ./logs/${universe}online_lr${online_lr}.log 2>&1 &
fi
echo $!