# Train
# mprompts=10
# nprompts=5
# lenprompts=5
# nohup python -u main.py --mprompts $mprompts --nprompts $nprompts --lenprompts $lenprompts > ./log/mprompts${mprompts}_nprompts${nprompts}_lenprompts${lenprompts}_lamb0.5.log 2>&1 &
# echo $!
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi
if [ ! -d "./backtest" ]; then
    mkdir ./backtest
fi
universe=csi500
mprompts=10
nprompts=5
lenprompts=5
online_lr=0.00005
only_backtest=true
sed -i "s/csi300\|csi500/$universe/g" workflow_config_master_Alpha158.yaml
if $only_backtest; then
    nohup python -u main.py --universe $universe --mprompts $mprompts --nprompts $nprompts --lenprompts $lenprompts --online_lr $online_lr --only_backtest > ./backtest/${universe}online_lr${online_lr}_mprompts${mprompts}_nprompts${nprompts}_lenprompts${lenprompts}_lamb0.5.log 2>&1 &
else
    nohup python -u main.py --universe $universe --mprompts $mprompts --nprompts $nprompts --lenprompts $lenprompts --online_lr $online_lr > ./logs/${universe}online_lr${online_lr}_mprompts${mprompts}_nprompts${nprompts}_lenprompts${lenprompts}_lamb0.5.log 2>&1 &
fi
echo $!