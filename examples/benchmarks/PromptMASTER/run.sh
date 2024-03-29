if [ ! -d "./logs" ]; then
    mkdir ./logs
fi
if [ ! -d "./backtest" ]; then
    mkdir ./backtest
fi
universe=csi300
mprompts=10
nprompts=5
lenprompts=5
only_backtest=true
sed -i "s/csi300\|csi500/$universe/g" workflow_config_master_Alpha158.yaml
if $only_backtest; then
    nohup python -u main.py --universe $universe --mprompts $mprompts --nprompts $nprompts --lenprompts $lenprompts --only_backtest > ./backtest/${universe}mprompts${mprompts}_nprompts${nprompts}_lenprompts${lenprompts}_lamb0.5.log 2>&1 &
else
    nohup python -u main.py --universe $universe --mprompts $mprompts --nprompts $nprompts --lenprompts $lenprompts  > ./logs/${universe}mprompts${mprompts}_nprompts${nprompts}_lenprompts${lenprompts}_lamb0.5.log 2>&1 &
fi
echo $!