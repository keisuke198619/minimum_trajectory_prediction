import glob, os, time, math, warnings, copy
import argparse
import random
import pickle

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["HDF5_USE_FILE_LOCKING"]="FALSE"

# customized ftns 
from rnn import load_model
from rnn.utils import num_trainable_params
from data_loader_soccer import Dataset
from preprocessing import *
from helpers import *
from sequencing import get_sequences

#from scipy import signal

# Keisuke Fujii, 2020
# modifying the codes
# https://github.com/samshipengs/Coordinated-Multi-Agent-Imitation-Learning
# https://github.com/ezhan94/multiagent-programmatic-supervision

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--n_GorS', type=int, required=True)
parser.add_argument('--n_roles', type=int, default=23)
parser.add_argument('--val_devide', type=int, default=10)
parser.add_argument('-t_step', '--totalTimeSteps', type=int, default=50)
parser.add_argument('--overlap', type=int, default=40)
parser.add_argument('--batchsize', type=int, required=True)
parser.add_argument('--n_epoch', type=int, required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('-ev_th','--event_threshold', type=int, default=50, help='event with frames less than the threshold will be removed')
parser.add_argument('--fs', type=int, default=10)
# parser.add_argument('--acc', type=int, default=0)
parser.add_argument('--cont', action='store_true')
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--numProcess', type=int, default=16)
parser.add_argument('--TEST', action='store_true')
parser.add_argument('--Challenge', action='store_true')
parser.add_argument('--Sanity', action='store_true')
parser.add_argument('--res', action='store_true')
parser.add_argument('--pretrain', type=int, default=0)
parser.add_argument('--finetune', action='store_true')
parser.add_argument('--drop_ind', action='store_true')
args, _ = parser.parse_known_args()

# directories
main_dir = '/home/s_dash/workspace6/RoboCup/' # './'
game_dir = main_dir+'data_'+args.data+'/'
print(game_dir+"-----------------------------------------------------")
args.game_dir = game_dir
Data = pd.read_pickle(game_dir+'train_data.pkl')
path_init = '/home/s_dash/workspace6/RoboCup/Code/minimum_trajectory_prediction/weights/' 
if args.Challenge:
    args.TEST = True

def run_epoch(train,rollout,hp):
    loader = train_loader if train == 1 else val_loader if train == 0 else test_loader

    losses = {} 
    losses2 = {}
    for batch_idx, (data) in enumerate(loader):
        # print(str(batch_idx))
        d1 = {'batch_idx': batch_idx}
        hp.update(d1)

        if args.cuda:
            data = data.cuda() #, data_y.cuda()
        # (batch, agents, time, feat) => (time, agents, batch, feat)
        data = data.permute(2, 1, 0, 3) #, data.transpose(0, 1)
        
        if train == 1:
            batch_losses, batch_losses2 = model(data, rollout, train, hp=hp)
            optimizer.zero_grad()
            total_loss = sum(batch_losses.values())
            total_loss.backward()
            if hp['model'] != 'RNN_ATTENTION': 
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        else:
            _, batch_losses, batch_losses2 = model.sample(data, rollout=True, burn_in=hp['burn_in'])# , TEST=TEST)
        
        for key in batch_losses:
            if batch_idx == 0:
                losses[key] = batch_losses[key].item() #numpy array -> key value is numpy array so we are able to use .item
            else:
                losses[key] += batch_losses[key].item()
        
        for key in batch_losses2:
            if batch_idx == 0:
                losses2[key] = batch_losses2[key].item()
            else:
                losses2[key] += batch_losses2[key].item()

    for key in losses:
        losses[key] /= len(loader.dataset)
    for key in losses2:
        losses2[key] /= len(loader.dataset)
    return losses, losses2

def loss_str(losses):
    ret = ''
    for key in losses:
        if 'L' in key and not 'mac' in key and not 'vel' in key:
            ret += ' {}: {:.0f} |'.format(key, losses[key])
        elif 'vel' in key:
            ret += ' {}: {:.3f} |'.format(key, losses[key])
        else: 
            ret += ' {}: {:.3f} |'.format(key, losses[key])
    return ret[:-2]

def run_sanity(args,game_files):
    for j in range(4): # why 4? due to permutation
        with open(game_files+str(j)+'.pkl', 'rb') as f:
            if j == 0:
                data = np.load(f,allow_pickle=True)[0]
            else:
                tmp = np.load(f,allow_pickle=True)[0] 
                data = np.concatenate([data,tmp],axis=1)            

    n_agents = args.n_agents
    batchSize,_,x_dim = data.shape
    n_feat = args.n_feat
    burn_in = args.burn_in
    fs = args.fs
    GT = data.copy()
    losses = {}
    losses['e_pos'] = np.zeros(batchSize)
    losses['e_vel'] = np.zeros(batchSize)
    losses['e_e_p'] = np.zeros(batchSize)
    losses['e_e_v'] = np.zeros(batchSize)

    for t in range(args.horizon):
        for i in range(n_agents):
            
            current_pos = data[:,t,n_feat*i+0:n_feat*i+2] # data why 3-dimension
            current_vel = data[:,burn_in,n_feat*i+2:n_feat*i+4]
            next_pos0 = GT[:,t+1,n_feat*i+0:n_feat*i+2]
            next_vel0 = GT[:,t+1,n_feat*i+2:n_feat*i+4]

            if t >= burn_in: 
                next_pos = current_pos + current_vel*fs      
                next_vel = current_vel 
                losses['e_pos'] += batch_error(next_pos, next_pos0)
                losses['e_vel'] += batch_error(next_vel, next_vel0)

                data[:,t+1,n_feat*i+0:n_feat*i+2] = next_pos
            if t == args.horizon-1:
                losses['e_e_p'] += batch_error(next_pos, next_pos0)
                losses['e_e_v'] += batch_error(next_vel, next_vel0)
                

    # del data
    losses['e_pos'] /= (args.horizon-burn_in)*n_agents 
    losses['e_vel'] /= (args.horizon-burn_in)*n_agents
    losses['e_e_p'] /= n_agents 
    losses['e_e_v'] /= n_agents

    avgL2_m = {}
    avgL2_sd = {}
    for key in losses:
        avgL2_m[key] = np.mean(losses[key])
        avgL2_sd[key] = np.std(losses[key])

    print('Velocity (Sanity Check)')
    print('(mean):'
        +' $' + '{:.2f}'.format(avgL2_m['e_pos'])+' \pm '+'{:.2f}'.format(avgL2_sd['e_pos'])+'$ &'
        +' $' + '{:.2f}'.format(avgL2_m['e_vel'])+' \pm '+'{:.2f}'.format(avgL2_sd['e_vel'])+'$ &'
        ) 
    print('(endpoint):'
        +' $' + '{:.2f}'.format(avgL2_m['e_e_p'])+' \pm '+'{:.2f}'.format(avgL2_sd['e_e_p'])+'$ &'
        +' $' + '{:.2f}'.format(avgL2_m['e_e_v'])+' \pm '+'{:.2f}'.format(avgL2_sd['e_e_v'])+'$ &'
        ) 
    
            
    losses['e_pos'] =  np.mean(losses['e_pos'])
    losses['e_e_p'] = np.mean(losses['e_e_p'])
    return losses

def batch_error(predict, true):
    error = np.sqrt(np.sum((predict[:,:2] - true[:,:2])**2,1))
    return error

def bound(val, lower, upper):
    """Clamps val between lower and upper."""
    if val < lower:
        return lower
    elif val > upper:
        return upper
    else:
        return val

if __name__ == '__main__':
    numProcess = args.numProcess  
    os.environ["OMP_NUM_THREADS"]=str(numProcess) 
    TEST = args.TEST

    # pre-process----------------------------------------------
    # acc = args.acc # output: 0: vel, 1: pos+vel, 2:vel+acc, 3: pos+vel+acc
    #args.hmm_iter = 500
    args.filter = True

    # all game ids file name, note that '/' or '\\' depends on the environment
    all_games_id = [i.split(os.sep)[-1].split('.')[0] for i in glob.glob(game_dir+'/*.pkl')]
    global fs
    fs = 1/args.fs
    if args.data == 'soccer':
        n_pl = 11
        subsample_factor = 10*fs        

    args.subsample_factor = subsample_factor
    event_threshold = args.event_threshold
    n_roles = args.n_roles
    n_GorS = args.n_GorS # games if NBA and seqs if soccer
    val_devide = args.val_devide
    batchSize = args.batchsize # 
    overlapWindow = args.overlap # 
    totalTimeSteps =  args.totalTimeSteps # 
    args.burn_in = 20 # int(totalTimeSteps/3)

    # save the processed file to disk to avoid repeated work
    game_file0 = '/home/s_dash/workspace6/RoboCup/Code/minimum_trajectory_prediction/data/all_'+args.data+'_games_'+str(n_GorS)+'_'+str(n_roles)

    game_file0 = game_file0 + '_filt'

    game_file0 = game_file0 + '/'
    if not os.path.isdir(game_file0):
        os.makedirs(game_file0)

    game_files_pre = game_file0 + '_pre'

    game_file0 = game_file0 + 'Fs' + str(args.fs) 

    game_file0 = game_file0 + '_' + str(batchSize) + '_' + str(totalTimeSteps)
    game_files = game_file0 
    game_files_val = game_file0 + '_val'+'.pkl'
    game_files_te = game_file0 + '_te'+'.pkl'    

    activeRoleInd = range(n_roles)
    activeRole = []; 
    activeRole.extend([str(n) for n in range(n_roles)]) # need to be reconsidered

    outputlen0 = 2
        
    numOfPrevSteps = 1 # We are only looking at the most recent character each time. 
    totalTimeSteps_test = totalTimeSteps
    n_feat = 4

    if os.path.isfile(game_files+'_te_0.pkl'): 
        print(game_files+'_te_0.pkl'+' can be loaded')
        with open(game_files+'_tr'+str(0)+'.pkl', 'rb') as f:
            X_all,len_seqs_val,len_seqs_test = np.load(f,allow_pickle=True)         

        print('load '+game_files+'_tr0.pkl')
    else:
        if os.path.isfile(game_files_pre+'.pkl'):
            print(game_files_pre+'.pkl will be loaded')
            with open(game_files_pre+'.pkl', 'rb') as f:
                game_data,game_data_te = np.load(f,allow_pickle=True)[:2] # ,_,_

        else: 
            print(game_files_pre+'.pkl is not existed then will be created')
            game_data,game_data_te,HSL_d,HSL_o = process_game_data(Data, all_games_id, args) 
            with open(game_files_pre+'.pkl', 'wb') as f:
                try: pickle.dump([game_data,game_data_te,HSL_d,HSL_o], f, protocol=4)
                except: import pdb; pdb.set_trace()

        print('Final number of events:', len(game_data), '+', len(game_data_te)) # 
        game_ind = np.arange(len(game_data))
        if args.data == 'soccer':
            game_train, game_test,_,_ = train_test_split(game_ind, game_ind, test_size=1/val_devide, random_state=42)
            game_data_te = [game_data[i] for i in game_test] 
            game_data = [game_data[i] for i in game_train] 

        # create sequences -----------------------------------------------------------
        X_train_all = get_sequences(game_data, activeRoleInd, 
                totalTimeSteps+5, overlapWindow, n_pl, n_feat) # [role][seqs][steps,feats]

        print('get train sequences')
        del game_data # -------------
        # split train/validation
        len_seqs = len(X_train_all) 
        X_ind = np.arange(len_seqs)
        ind_train, ind_val,_,_ = train_test_split(X_ind, X_ind, test_size=1/val_devide, random_state=42)

        featurelen = X_train_all[0].shape[1] 
        len_seqs_tr = len(ind_train)
        offSet_tr = math.floor(len_seqs_tr / batchSize)
        batchSize_val = len(ind_val)

        X_all = np.zeros([len(ind_train), totalTimeSteps+4, featurelen])
        X_val_all = np.zeros([len(ind_val), totalTimeSteps+4, featurelen])
        # for i, X_train in enumerate(X_train_all):
        i_tr = 0; i_val = 0
        for b in range(len_seqs):  
            if set([b]).issubset(set(ind_train)):
                for r in range(totalTimeSteps+4):
                    X_all[i_tr][r][:] = np.squeeze(X_train_all[b][r,:])
                i_tr += 1
            else:
                for r in range(totalTimeSteps+4):
                    X_val_all[i_val][r][:] = np.squeeze(X_train_all[b][r,:])
                i_val += 1

        print('create train sequences')
        
        del X_train_all

        # for test data-------------
        X_test_all = get_sequences(game_data_te, activeRoleInd, 
            totalTimeSteps+5, overlapWindow, n_pl, n_feat) # [role][seqs][steps,feats]
        del game_data_te

        len_seqs_val = len(X_val_all)  
        len_seqs_test = len(X_test_all)  
        batchSize_test = len_seqs_test # args.batchsize # 32
        len_seqs_test0 = len_seqs_test
        ind_test = np.arange(len_seqs_test)

        X_test_test_all = np.zeros([len_seqs_test, totalTimeSteps_test+4, featurelen]) 
        i_te = 0
        for b in range(len_seqs_test0):
            if args.data == 'soccer':
                for r in range(totalTimeSteps_test+4):
                    X_test_test_all[b][r][:] = np.squeeze(X_test_all[b][r,:])

        print('create test sequences')
        # if offSet_tr > 0: 
        for j in range(offSet_tr):
            tmp_data = X_all[j*batchSize:(j+1)*batchSize,:,:]
            with open(game_files+'_tr'+str(j)+'.pkl', 'wb') as f:
                pickle.dump([tmp_data,len_seqs_val,len_seqs_test], f, protocol=4) 

        J = 8 #why 8
        batchval = int(len_seqs_val/J)
        for j in range(J):
            if j < J-1:
                tmp_data = X_val_all[j*batchval:(j+1)*batchval,:,:]
            else:
                tmp_data = X_val_all[j*batchval:,:,:]
            with open(game_files+'_val_'+str(j)+'.pkl', 'wb') as f:
                pickle.dump([tmp_data], f, protocol=4)                       

        batchte = int(len_seqs_test/J)
        for j in range(J):
            if j < J-1:
                tmp_data = X_test_test_all[j*batchte:(j+1)*batchte,:,:]
            else:
                tmp_data = X_test_test_all[j*batchte:,:,:]
            with open(game_files+'_te_'+str(j)+'.pkl', 'wb') as f:
                pickle.dump([tmp_data], f, protocol=4)     

        if True:
            experiment_path = '/home/s_dash/workspace6/RoboCup/Code/minimum_trajectory_prediction/test_samples/gt_'+str(n_roles)
            experiment_path2 = '/home/s_dash/workspace6/RoboCup/Code/minimum_trajectory_prediction/test_samples/input_'+str(n_roles)
            if not os.path.exists(experiment_path):
                os.makedirs(experiment_path)
                os.makedirs(experiment_path2)

            # Save samples to CSV        
            for seq in range(X_test_test_all.shape[0]):
                sample_ = X_test_test_all[seq].reshape((-1,23,n_feat))[:totalTimeSteps,:,:2] #
                # ground truth
                sample_path = os.path.join(experiment_path,str(seq)+'.csv')
                # Generate the column names without list comprehension
                columns = []
                for agent in range(sample_.shape[1]):  # Loop over agents
                    for coord in ['x', 'y']:          # Loop over coordinates
                        columns.append("agent_{}_{}".format(agent, coord))

                # Create the DataFrame
                df = pd.DataFrame(
                    sample_.reshape(sample_.shape[0], -1), 
                    columns=columns
                )
                df.to_csv(sample_path, index_label='time')
                # test input
                sample_path = os.path.join(experiment_path2,str(seq)+'.csv')
                sample_ = sample_[:args.burn_in]
                df = pd.DataFrame(sample_.reshape(sample_.shape[0], -1), columns=["agent_{}_{}".format(agent, coord) for agent in range(sample_.shape[1]) for coord in ['x', 'y']])

                #df = pd.DataFrame()
                #for t in range(args.burn_in):
                #    for agent in range(sample_.shape[0]):
                #        df.loc[t, f'agent_{agent}_x'] = sample_[agent, t, 0]
                #        df.loc[t, f'agent_{agent}_y'] = sample_[agent, t, 1]
                df.to_csv(sample_path, index_label='time')
            print('Samples saved to {}'.format(sample_path))
             
        del X_val_all, X_test_test_all, tmp_data

        print('save train and test sequences')
        with open(game_files+'_tr'+str(0)+'.pkl', 'rb') as f:
            X_all,len_seqs_val,len_seqs_test = np.load(f,allow_pickle=True) 

     

    # count batches 
    offSet_tr =  len(glob.glob(game_files+'_tr*.pkl'))
    # variables
    featurelen = X_all.shape[2] #[0][0][0]#see get_sequences in sequencing.py
    len_seqs_tr = batchSize*offSet_tr
    print('featurelen: '+str(featurelen)+' train_seqs: '+str(len_seqs_tr)+' val_seqs: '+str(len_seqs_val)+' test_seqs: '+str(len_seqs_test))
    
    # parameters for RNN -----------------------------------
    init_filename0 = path_init+ 'sub' + str(args.fs) + '_' + str(n_roles)
    init_filename0 = init_filename0 + 'filt_'  
   

    init_filename0 = init_filename0 + args.model + '_' + args.data + '/'
    init_filename0 = init_filename0 + str(batchSize) + '_' + str(totalTimeSteps)      
    if args.drop_ind:
        init_filename0 = init_filename0 + '_drop_ind' 

    if args.res:
        init_filename0 = init_filename0 + '_res' 

    if not os.path.isdir(init_filename0):
        os.makedirs(init_filename0)
    init_pthname = '{}_state_dict'.format(init_filename0)
    print('model: '+init_filename0)

    if not os.path.isdir(init_pthname):
        os.makedirs(init_pthname)

    if (args.n_GorS==7500 and args.data == 'soccer'):
        batchSize = int(batchSize/2)
    # args.hard_only = True
    args.dataset = args.data
    args.n_feat = n_feat
    args.fs = fs
    args.game_files = game_files  
    args.game_files_val = game_files_val
    args.game_files_te = game_files_te
    args.start_lr = 1e-3 
    args.min_lr = 1e-3 
    clip = True # gradient clipping
    args.seed = 200
    save_every = 1
    args.batch_size = batchSize
    # args.cont = False # continue training previous best model
    args.x_dim = outputlen0 # output
    args.y_dim = featurelen # input
    args.z_dim = 64 
    args.h_dim = 64 #128 
    args.rnn_dim = 100 # 100
    args.n_layers = 2
    args.rnn_micro_dim = args.rnn_dim

    args.horizon = totalTimeSteps
    args.n_agents = len(activeRole)
    args.n_all_agents = 22 if args.data == 'soccer' else 10 
    if not torch.cuda.is_available():
        args.cuda = False
        print('cuda is not used')
    else:
        args.cuda = True
        print('cuda is used')
    ball_dim = 4 
    # Parameters to save
    temperature = 1 if args.data == 'soccer' else 1 
        
    params = {
        'model' : args.model,
        'res' : args.res,
        'dataset' : args.dataset,
        'x_dim' : args.x_dim,
        'y_dim' : args.y_dim,
        'z_dim' : args.z_dim,
        'h_dim' : args.h_dim,
        'rnn_dim' : args.rnn_dim,
        'n_layers' : args.n_layers, 
        'len_seq' : totalTimeSteps,  
        'n_agents' : args.n_agents,    
        'min_lr' : args.min_lr,
        'start_lr' : args.start_lr,
        'seed' : args.seed,
        'cuda' : args.cuda,
        'n_feat' : n_feat,
        'fs' : fs,
        'embed_size' : 32, # 8
        'embed_ball_size' : 32, # 8
        'burn_in' : args.burn_in,
        'horizon' : args.horizon,
        'rnn_micro_dim' : args.rnn_micro_dim,
        'ball_dim' : ball_dim,
        'n_all_agents' : args.n_all_agents,
        'temperature' : temperature,
        'drop_ind' : args.drop_ind,
    }
        
    #'pretrain' : args.pretrain,
        
    # Set manual seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    ####### Sanity check ##################
    if args.Sanity:
        losses = run_sanity(args,game_files+'_te_')

    # Load model
    
    model = load_model(args.model, params, parser)

    if args.cuda:
        model.cuda()
    # Update params with model parameters
    params = model.params
    params['total_params'] = num_trainable_params(model)

    # Create save path and saving parameters
    pickle.dump(params, open(init_filename0+'/params.p', 'wb'), protocol=2)

    # Continue a previous experiment, or start a new one
    if args.cont:
        if os.path.exists('{}_best.pth'.format(init_pthname)): 
            # state_dict = torch.load('{}_12.pth'.format(init_pthname))
            state_dict = torch.load('{}_best.pth'.format(init_pthname))
            model.load_state_dict(state_dict)
            print('best model was loaded')
        else:
            print('args.cont = True but file did not exist')


    print('############################################################')

    # Dataset loaders
    num_workers = args.num_workers
    kwargs = {'num_workers': num_workers, 'pin_memory': True} if args.cuda else {}
    kwargs2 = {'num_workers': num_workers, 'pin_memory': True} if args.cuda else {}
    print('num_workers:'+str(num_workers))
    batchSize_val = len_seqs_val if len_seqs_val <= batchSize else batchSize
    batchSize_test = len_seqs_test if len_seqs_test <= int(batchSize/2) else batchSize # int(/4)
    if (args.n_GorS==7500 and args.dataset == 'soccer'):
        batchSize_val = int(batchSize/4*3)
        batchSize_test = 128

    if not TEST:    
        train_loader = DataLoader(
            Dataset(args, len_seqs_tr, train=1),
            batch_size=batchSize, shuffle=False, **kwargs)    
        val_loader = DataLoader(
            Dataset(args, len_seqs_val, train=0),
            batch_size=batchSize_val, shuffle=False, **kwargs2)
    test_loader = DataLoader(
        Dataset(args, len_seqs_test, train=-1),
        batch_size=batchSize_test, shuffle=False, **kwargs2)
    print('batch train: '+str(batchSize)+' val:'+str(batchSize_val)+' test: '+str(batchSize_test))

    ###### TRAIN LOOP ##############
    best_val_loss = 0
    epochs_since_best = 0
    lr = max(args.start_lr, args.min_lr) # if not args.finetune else 1e-4
    epoch_first_best = -1
    #print('epoch_first_best: '+str(epoch_first_best))

    pretrain_time =  0
    
    # hyperparams = {'model': args.model,'acc': acc,'burn_in': args.burn_in}
    hyperparams = {'model': args.model,'burn_in': args.horizon, 'pretrain':(0 < pretrain_time)}
    
    if not TEST:
        for e in range(args.n_epoch):
            epoch = e+1
            print('epoch '+str(epoch))
            pretrain = (epoch <= pretrain_time)
            hyperparams['pretrain'] = pretrain

            # Set a custom learning rate schedule
            if epochs_since_best == 5: # and lr > args.min_lr:
                # Load previous best model
                filename = '{}_best.pth'.format(init_pthname)

                state_dict = torch.load(filename)

                # Decrease learning rate
                # lr = max(lr/3, args.min_lr)
                # print('########## lr {} ##########'.format(lr))
                epochs_since_best = 0
            else:
                if not hyperparams['pretrain'] and not args.finetune:
                    # lr = lr*0.99 # 9
                    print('########## lr {:.4e} ##########'.format(lr)) 
                    epochs_since_best += 1
                

            optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=lr)
            
            start_time = time.time()
            
            print('pretrain:'+str(hyperparams['pretrain']))
            hyperparams['burn_in'] = args.horizon
            train_loss,train_loss2 = run_epoch(train=1, rollout=False, hp=hyperparams)
            print('Train:\t'+loss_str(train_loss)+'|'+loss_str(train_loss2))
            
            hyperparams['burn_in'] = args.burn_in
            val_loss,val_loss2 = run_epoch(train=0, rollout=True, hp=hyperparams)
            print('RO Val:\t'+loss_str(val_loss)+'|'+loss_str(val_loss2))

            total_val_loss = sum(val_loss.values())

            epoch_time = time.time() - start_time
            print('Time:\t {:.3f}'.format(epoch_time))

            # Best model on test set
            if e > epoch_first_best and (best_val_loss == 0 or total_val_loss < best_val_loss): 
                best_val_loss_prev = best_val_loss
                best_val_loss = total_val_loss
                epochs_since_best = 0

                filename = '{}_best.pth'.format(init_pthname)

                torch.save(model.state_dict(), filename)
                print('##### Best model #####')
                if epoch > pretrain_time and (best_val_loss_prev-best_val_loss)/best_val_loss < 0.0001 and best_val_loss_prev != 0:
                    print('best loss - current loss: ' +str(best_val_loss_prev)+' - '+str(best_val_loss))
                    break 


            # Periodically save model
            if epoch % save_every == 0:
                filename = '{}_{}.pth'.format(init_pthname, epoch)
                torch.save(model.state_dict(), filename)
                print('########## Saved model ##########')

                           
        print('Best Val Loss: {:.4f}'.format(best_val_loss))
    
    # Load params
    params = pickle.load(open(init_filename0+'/params.p', 'rb'))
    
    # Load model
    state_dict = torch.load('{}_best.pth'.format(init_pthname, params['model']), map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)

    # Load ground-truth states from test set
    loader = test_loader 

    if True:
        print('test sample')
        # Sample trajectory
        samples = np.zeros((args.horizon,args.n_agents,len_seqs_test,featurelen)) 
        losses = {}
        losses2 = {}

        start_time = time.time()
        i = 0
        for batch_idx, (data) in enumerate(loader):
            if args.cuda:
                data = data.cuda() #, data_y.cuda()
                # (batch, agents, time, feat) => (time, agents, batch, feat) 
            data = data.permute(2, 1, 0, 3)
            
            sample, output, output2 = model.sample(data, rollout=True, burn_in=args.burn_in, n_sample=1, TEST = True, Challenge = args.Challenge)

            samples[:,:,batch_idx*batchSize_test:(batch_idx+1)*batchSize_test] = sample.detach().cpu().numpy()[:-4]

            del sample 
            if not args.Challenge:
                for key in output:
                    if batch_idx == 0:
                        losses[key] = np.zeros(1)
                        losses2[key] = np.zeros((len_seqs_test))
                    losses[key] += np.sum(output[key].detach().cpu().numpy(),axis=1)
                    try: losses2[key][batch_idx*batchSize_test:(batch_idx+1)*batchSize_test] = output[key].detach().cpu().numpy()
                    except: import pdb; pdb.set_trace()
                    
                for key in output2:
                    if batch_idx == 0:
                        losses[key] = np.zeros(1)
                        losses2[key] = np.zeros((len_seqs_test))
                    losses[key] += np.sum(output2[key].detach().cpu().numpy(),axis=1)
                    losses2[key][batch_idx*batchSize_test:(batch_idx+1)*batchSize_test] = output2[key].detach().cpu().numpy()


        epoch_time = time.time() - start_time
        print('Time:\t {:.3f}'.format(epoch_time)) # Sample {} r*n_smp_b,
            
        if not args.Challenge: # create Mean + SD Tex Table for positions------------------------------------------------
            avgL2_m = {}
            avgL2_sd = {}
            for key in losses2:
                avgL2_m[key] =  np.mean(losses2[key])
                avgL2_sd[key] = np.std(losses2[key])

            print(args.model)
            print('(mean):'
                +' $' + '{:.2f}'.format(avgL2_m['e_pos'])+' \pm '+'{:.2f}'.format(avgL2_sd['e_pos'])+'$ &'
                +' $' + '{:.2f}'.format(avgL2_m['e_vel'])+' \pm '+'{:.2f}'.format(avgL2_sd['e_vel'])+'$ &'
                ) 
            print('(endpoint):'
                +' $' + '{:.2f}'.format(avgL2_m['e_e_p'])+' \pm '+'{:.2f}'.format(avgL2_sd['e_e_p'])+'$ &'
                +' $' + '{:.2f}'.format(avgL2_m['e_e_v'])+' \pm '+'{:.2f}'.format(avgL2_sd['e_e_v'])+'$ &'
                ) 
        else: # challenge   
            # Save samples
            experiment_path = '/home/s_dash/workspace6/RoboCup/Code/minimum_trajectory_prediction/results/test_'+str(n_roles)+'/submission'
            if not os.path.exists(experiment_path):
                os.makedirs(experiment_path)

            # Save samples to CSV        
            for seq in range(samples.shape[2]):
                sample_ = samples[:, 0, seq].reshape((-1,23,n_feat))[:,:,:2] #
                sample_path = os.path.join(experiment_path, str(seq)+'.csv')
                df = pd.DataFrame(sample_.reshape(sample_.shape[0], -1), columns=["agent_{}_{}".format(agent, coord) for agent in range(sample_.shape[1]) for coord in ['x', 'y']])
                df.to_csv(sample_path, index_label='time')
            print('Samples saved to {}'.format(experiment_path))
        import pdb; pdb.set_trace()
