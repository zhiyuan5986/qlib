
# git clone https://github.com/SJTU-Quant/qlib.git
# cd qlib
# python setup.py develop

# cd examples/benchmarks/Graphs_dgl

echo "=============================A_share, stock-stock, RSR, heterograph============================="
python run_ts_dgl.py --data_root "/home/zhangzexi/.qlib/qlib_data/" \
                     --ckpt_root "/data/dengjiale/qlib_exp/tmp_ckpt/" \
                     --result_root "/data/dengjiale/qlib_exp/results/" \
                     --market "A_share" \
                     --relation_type "stock-stock" \
                     --graph_model "RSR" \
                     --graph_type "heterograph" \
                     --gpu 0

echo "=============================A_share, stock-stock, simpleHGN, heterograph============================="
python run_ts_dgl.py --data_root "/home/zhangzexi/.qlib/qlib_data/" \
                     --ckpt_root "/data/dengjiale/qlib_exp/tmp_ckpt/" \
                     --result_root "/data/dengjiale/qlib_exp/results/" \
                     --market "A_share" \
                     --relation_type "stock-stock" \
                     --graph_model "simpleHGN" \
                     --graph_type "heterograph" \
                     --gpu 2

echo "=============================A_share, stock-stock, GAT, homograph============================="
python run_ts_dgl.py --data_root "/home/zhangzexi/.qlib/qlib_data/" \
                     --ckpt_root "/data/dengjiale/qlib_exp/tmp_ckpt/" \
                     --result_root "/data/dengjiale/qlib_exp/results/" \
                     --market "A_share" \
                     --relation_type "stock-stock" \
                     --graph_model "GAT" \
                     --graph_type "homograph" \
                     --gpu 3