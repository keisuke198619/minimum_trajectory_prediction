#!/bin/bash
# a simple demostration code with small samples
# srun -p ubuntu -w nsx -c 8 --gres=gpu:1 --pty bash &> logs.txt
# cd /home/s_dash/workspace6/RoboCup/Code/minimum_trajectory_prediction
# pip install -r requirements.txt # install packages
source $HOME/workspace6/RoboCup/bin/activate

python3 main.py --data soccer --n_GorS 250 --batchsize 64 --n_epoch 10 --model RNN_GAUSS --num_workers 0 --Sanity 
# --TEST --Challenge
python evaluation.py --submit ./results/test_23_new/submission --gt ./test_samples/gt_23 
