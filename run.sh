#!/bin/bash
# a simple demostration code with small samples
srun -p ubuntu -w nsx -c 8 --gres=gpu:1 --pty bash
cd /home/fujii/workspace5/work/minimum_trajectory_prediction
source $HOME/workspace4/virtualenvs/cause38/bin/activate 

# python main_stats.py --data soccer --n_GorS 250 --batchsize 64 --n_epoch 10 --model RNN_GAUSS --num_workers 0 --Sanity # --TEST --Challenge
# python evaluation.py --submit ./results/test_23/submission --gt ./test_samples/gt_23 

python download.py --subpaths rc2021-roundrobin/normal/alice2021-helios2021 rc2021-roundrobin/normal/alice2021-hfutengine2021
python main.py -t_step 50 --batchsize 64 --n_epoch 10 --model RNN_GAUSS --num_workers 0 --Sanity # --TEST
python main.py -t_step 50 --batchsize 64 --n_epoch 10 --model RNN_GAUSS --num_workers 0 --challenge_data ./test_samples/input
# python evaluation.py --submit ./results/test/submission --gt ./test_samples/gt --input ./test_samples/input 