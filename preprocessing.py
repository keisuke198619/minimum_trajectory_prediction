# preprocessing.py
import glob, os, sys, math, warnings, copy, time
import numpy as np
import pandas as pd
from scipy import signal

from features import create_static_features, create_dynamic_features, flatten_moments_soccer
from sequencing import subsample_sequence

# Keisuke Fujii, 2020
# modifying the code https://github.com/samshipengs/Coordinated-Multi-Agent-Imitation-Learning

# ================================================================================================
# remove_non_eleven ==============================================================================
# ================================================================================================
def remove_non_eleven(events_df, event_length_th,n_pl, verbose=False):
    df = events_df.copy() # shape [frames x 8 columns] 
    # playbyplay moments  visitor orig_events  start_time_left home  quarter  end_time_left  
    home_id = []
    away_id = []

    def remove_non_eleven_(moments, event_length_th,n_pl, verbose=False):
        segments = []
        segment = []
        # looping through each moment
        for i in range(len(moments)):
            # get moment dimension
            moment_dim = len(moments[i][0]) # 46-dims
            accurate_dim = 46
            
            if moment_dim == accurate_dim: 
                segment.append(moments[i]) # less than ten players or basketball is not on the court
                
            else:
                # only grab these satisfy the length threshold
                if len(segment) >= event_length_th:
                    segments.append(segment)
                # reset the segment to empty list
                segment = []

        # grab the last one
        if len(segment) >= event_length_th:
            segments.append(segment)
        if False: # len(segments) == 0:
            print('Warning: Zero length event returned')
        return segments
    # process for each event (row)
    df['chunked_moments'] = df.moments.apply(lambda m: remove_non_eleven_(m, event_length_th, n_pl, verbose))
    # in case there's zero length event
    df = df[df['chunked_moments'].apply(lambda e: len(e)) != 0]
    df['chunked_moments'] = df['chunked_moments'].apply(lambda e: e[0])
    return df['chunked_moments'].values, {'home_id': home_id, 'away_id': away_id}

# ================================================================================================
# filters ================================================================================
# ================================================================================================
def filters(events_df,fs):
    order = 2 # order of the filter
    Nq = 1/(2*fs) # Nyquist frequency (Hz)  
    fp = 2 # low pass frequency (Hz)         
    b, a = signal.butter(order, fp/Nq, 'low', analog=False)

    df = events_df.copy()
    data_list = []

    for m in df.moments: # moments[seq][time][feature]
        data0 = np.zeros((len(m),len(m[0]))) # time, feature
        for i in range(len(m)): # time length
            data0[i,:] = m[i]
        
        data_filt = signal.filtfilt(b, a, data0, axis=0) 

        data_list0 = []
        for i in range(len(m)): # time length
            data_list0.append(data_filt[i,:])  
        data_list.append(data_list0)    

    return data_list 

def process_game_data(Data, game_ids, args): # event_threshold, subsample_factor,dataset,n_roles):
    def process_game_data_(game_id, events_df, args):
        event_threshold = args.event_threshold
        dataset = args.data
        filter = args.filter
            
        if dataset == 'soccer':
            n_pl = 11
            fs = 1/10.

        # remove non elevens
        result, _ = remove_non_eleven(events_df, event_threshold,n_pl)
        df = pd.DataFrame({'moments': result}) # list: maybe segments*frames*data (e.g. 263*150*6)
        df_tr = df


        # features 
        # flatten data
        result_tr, _ = flatten_moments_soccer(df_tr)  
        df_tr = pd.DataFrame({'moments': result_tr})  # list: [seqs][t][46-dim]

        # filter
        if filter:
            result_tr = filters(df_tr,fs)
            df_tr = pd.DataFrame({'moments': result_tr})

        # static features
        result_tr = create_static_features(df_tr,n_pl)
        df_tr = pd.DataFrame({'moments': result_tr})

        # dynamic features
        result_tr = create_dynamic_features(df_tr, fs, n_pl)
        df_tr = pd.DataFrame({'moments': result_tr}) 

        return df_tr

    game_tr = [] 
    game_te = []
    subsample_factor = args.subsample_factor
    dataset = args.data

    if dataset == 'soccer':
        n_pl = 11
        data_unit = 'datasets'
        iter = len(game_ids)
    
    for i in range(iter):
        print('working on game {0:} | {1:} out of total {2:} {3:}'.format(game_ids[i], i+1, iter,data_unit)) # len(game_ids)
        game_data = pd.read_pickle(args.game_dir+game_ids[i]+'.pkl')

        if dataset == 'soccer':
            data_dict = {}
            data_dict = {'events':[]}
            if 'train_data' in game_ids[i]:
                len_seqs = args.n_GorS 
            else: 
                len_seqs = len(game_data)
            for j in range(len_seqs): 
                data_list = [[] for _ in range(game_data["sequence_%d"%(j+1)].shape[0])]
                for t in range(game_data["sequence_%d"%(j+1)].shape[0]): 
                    data_list[t].append(game_data["sequence_%d"%(j+1)][t]) # 46-dim

                data_dict2 = {}
                data_dict2 = {'moments':data_list}
                data_dict['events'].append(data_dict2)
                
            events_df = pd.DataFrame(data_dict['events']) # events_df.moments[seqs][t][46-dim]

        if dataset == 'soccer': 
            if 'train_data' in game_ids[i]:
                game_tr = process_game_data_(game_ids[i], events_df, args)
            elif 'test_data' in game_ids[i]:
                game_te0 = process_game_data_(game_ids[i], events_df, args)
                game_te.append(game_te0)

    if dataset == 'soccer': 
        df_tr = game_tr 
        df_te = pd.concat(game_te, axis=0)
        
    # subsample
    result = subsample_sequence(df_tr.moments.values, subsample_factor) # [seqs]frames*features 
    #print(result[0][0].shape) # ndarray: [seqs][frames][features] 
    result_te = subsample_sequence(df_te.moments.values, subsample_factor) #  
    return result, result_te, None, None