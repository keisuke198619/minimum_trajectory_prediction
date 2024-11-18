# sequencing.py
import glob, os, sys, math, warnings, copy, time
import numpy as np
import pandas as pd
from scipy import signal
# Keisuke Fujii, 2020
# modifying the code https://github.com/samshipengs/Coordinated-Multi-Agent-Imitation-Learning

# ===============================================================================
# subsample_sequence ============================================================
# ===============================================================================
def subsample_sequence(events, subsample_factor, random_sample=False):
    if subsample_factor == 0 or round(subsample_factor*10)==10:
        return events
    
    def subsample_sequence_(moments, subsample_factor, random_sample=False):#random_state=42):
        ''' 
            moments: a list of moment 
            subsample_factor: number of folds less than orginal
            random_sample: if true then sample a random one from the window of subsample_factor size
        '''
        seqs = np.copy(moments)
        moments_len = seqs.shape[0]
        if subsample_factor > 0:
            n_intervals = moments_len//subsample_factor # number of subsampling intervals
        else: 

            n_intervals = int(moments_len//-subsample_factor)

        left = moments_len % subsample_factor # reminder

        if random_sample:
            if left != 0:
                rs = [np.random.randint(0, subsample_factor) for _ in range(n_intervals)] + [np.random.randint(0, left)]
            else:
                rs = [np.random.randint(0, subsample_factor) for _ in range(n_intervals)]
            interval_ind = range(0, moments_len, subsample_factor)
            # the final random index relative to the input
            rs_ind = np.array([rs[i] + interval_ind[i] for i in range(len(rs))])
            return seqs[rs_ind, :]
        else:
            if round(subsample_factor*10) == round(subsample_factor)*10: # int
                s_ind = np.arange(0, moments_len, subsample_factor)
                return seqs[s_ind, :]
            else:
                # only when 10 Hz undersampling in NBA (25 Hz)
                if round(subsample_factor*10) == 25:
                    up = 2
                    down = 5
                seqs2 = signal.resample_poly(seqs, up, down, axis=0, padtype='line')
                seqs2 = seqs2[1:-1]

                return seqs2
                          
    return [subsample_sequence_(ms, subsample_factor) for ms in events]



def get_sequences(single_game, policy, sequence_length, overlap, n_pl, n_feat):
    ''' create events where each event is a list of sequences from
        single_game with required sequence_legnth and overlap

        single_game: A list of events
        sequence_length: the desired length of each event (a sequence of moments)
        overlap: how much overlap wanted for the sequence generation
 
    '''

    X_all = []
    Y_all = []   

    npl = n_pl*2
    index0 = np.array(range(single_game[0].shape[1])).astype(int) # length of features

    for p in policy:
        X = []
        Y = []
        # create index
        index = [] 
        if n_pl == 11:
            for pl in range(npl):
                index = np.append(index,index0[pl*2:pl*2+2]) # positions 3-4
                index = np.append(index,index0[2204+pl*2:2204+pl*2+2]) # velocities 5-6
            
            index = np.append(index,index0[44:46]) # ball positions
            index = np.append(index,index0[2202:2204]) # ball velocity

        index = index.astype(int)
        #index = np.array([p*2,p*2+1, \
        #    25+p,35+p,45+p,55+p,65+p,75+p,85+p,95+p,\
        #    p*2+105,p*2+106])
        for i in single_game:
            i_len = len(i)
            i2 = np.array(i) # copy
            sequence0 = np.zeros((i_len,index.shape[0]))
            
            for t in range(i_len):
                sequence0[t,:] = i2[t,index].T
            
            # create sequences
            if i_len >= sequence_length:
                sequences0 = [sequence0[-sequence_length:,:] if j + sequence_length > i_len-1 else sequence0[j:j+sequence_length,:] \
                    for j in range(0, i_len-overlap, sequence_length-overlap)] # for the states
                #sequences = [np.array(i[-sequence_length:]) if j + sequence_length > i_len-1 else np.array(i[j:j+sequence_length]) \
                #    for j in range(0, i_len-overlap, sequence_length-overlap)] # for the actions     

                state = [np.roll(kk, -1, axis=0)[:-1, :] for kk in sequences0] # state: drop the last row as the rolled-back is not real
                
                action = [np.roll(kk[:, p*n_feat+3:p*n_feat+9], -1, axis=0)[:-1, :] for kk in sequences0] 
                # sequences = [l[:-1, :] for l in sequences] # since target has dropped one then sequence also drop one
                X += state  
                Y += action  
        X_all.append(X) 
        Y_all.append(Y) 
    return X_all, Y_all

