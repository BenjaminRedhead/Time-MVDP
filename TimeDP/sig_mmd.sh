source /workspace/writeable/SigRisk/.venv/bin/activate
cd /workspace/TimeCraft/TimeDP

python evaluate_sig_mmd.py \
    --results_dir /workspace/writeable/TimeCraft/TimeDP/multivariate/ETTh1/ \
    --dataset_csv /workspace/TimeCraft/TimeDP/data/ETTh1/ETTh1.csv