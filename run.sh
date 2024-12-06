#!/bin/bash
# a simple demostration code with small samples
srun -p ubuntu -w nsx -c 8 --gres=gpu:1 --pty bash
cd /home/fujii/workspace5/work/minimum_trajectory_prediction
source $HOME/workspace4/virtualenvs/cause38/bin/activate 

# python main_stats.py --data soccer --n_GorS 250 --batchsize 64 --n_epoch 10 --model RNN_GAUSS --num_workers 0 --Sanity # --TEST --Challenge
# python evaluation.py --submit ./results/test_23/submission --gt ./test_samples/gt_23 

python download_to_pickle.py
python main.py --n_GorS 1 --batchsize 64 --n_epoch 10 --model RNN_GAUSS --num_workers 0 --Sanity # --TEST --Challenge
python evaluation.py # --submit ./results/test_23/submission --gt ./test_samples/gt_23 