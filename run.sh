#!/bin/bash
# a simple demostration code with small samples
srun -p ubuntu -w nsx -c 8 --gres=gpu:1 --pty bash
cd /home/fujii/workspace5/work/minimum_traj_pred
source $HOME/workspace4/virtualenvs/cause38/bin/activate 

python -u main.py --data soccer --n_GorS 250 --n_roles 10 --batchsize 64 --n_epoch 1 -ev_th 50 --model RNN_GAUSS --num_workers 0 --Sanity
