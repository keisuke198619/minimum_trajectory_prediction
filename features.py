# features.py
import glob, os, sys, math, warnings, copy, time, glob
import numpy as np
import pandas as pd

# Keisuke Fujii, 2020
# modifying the code https://github.com/samshipengs/Coordinated-Multi-Agent-Imitation-Learning

def flatten_moments_soccer(events_df):
    ''' This changes the nested list that represents single frame 
        to a 1-D array.
     '''
    df = events_df.copy()
    team_id = []
    def flatten_moment(moment):
        m = np.array(moment[0])
        # the defending team is always the first 
        # goalkeeper is the first player but for processing, is moved to the last
        features = np.concatenate((m[2:22],m[0:2], # x,y of 11 defenders
                                   m[24:44],m[22:24],    # x,y of 11 attackers
                                   m[44:46]))             # ball x,y

        return features
        
    df['flattened'] = df.moments.apply(lambda ms: [flatten_moment(m) for m in ms])                                     
    return df['flattened'].values, team_id

# =================================================================
# create_static_features ==========================================
# =================================================================
def create_static_features(events_df,n_pl):

    df = events_df.copy()
    def create_static_features_(moment,n_pl):
        # distance of each players to the ball
        player_xy = moment[:n_pl*4]
        b_xy = moment[n_pl*4:n_pl*4+2]
        if n_pl == 5:
            hoop_xy = np.array([3.917, 25])
        elif n_pl == 11:
            hoop_xy = np.array([52.5,0])
        
        def disp_(pxy, target, n_pl):
            # dispacement to ball or goal: -pi:piz
            disp = pxy.reshape(-1, 2) - np.tile(target, (n_pl*2, 1))
            assert disp.shape[0] == n_pl*2
            r = np.sqrt(disp[:,0]**2 + disp[:, 1]**2)  
            cos_theta = np.zeros(disp.shape[0])
            sin_theta = np.zeros(disp.shape[0])
            theta = np.zeros(disp.shape[0])
            
            for i in range(disp.shape[0]):
                if r[i] != 0:
                    cos_theta[i] = disp[i, 0]/r[i] # costheta
                    sin_theta[i] = disp[i, 1]/r[i] # sintheta
                    theta[i] = np.arccos(cos_theta[i]) # theta
            return np.concatenate((r, cos_theta, sin_theta, theta))

        moment = np.concatenate((moment, disp_(player_xy, b_xy, n_pl), disp_(player_xy, hoop_xy, n_pl)))
        for pl in range(n_pl*2): # relationship between all players and defenders => all players
            player2_xy = moment[pl*2:pl*2+2]
            moment = np.concatenate((moment, disp_(player_xy, player2_xy, n_pl)))
        return moment
    # vertical stack s.t. now each event i.e. a list of moments becomes an array
    # where each row is a frame (moment)
    df['enriched'] = df.moments.apply(lambda ms: np.vstack([create_static_features_(m,n_pl) for m in ms]))
    return df['enriched'].values


# =================================================================
# create_dynamic_features =========================================
# =================================================================
def create_dynamic_features(events_df, fs, n_pl):
    df = events_df.copy()
    def create_dynamic_features_(moments, fs, n_pl):
        ''' moments: (moments length, n existing features)'''
        ball_dim = 3 if n_pl == 5 else 2
        pxy = moments[:, :n_pl*4+ball_dim] # get the players x,y and basketball x,y,z coordinates
        next_pxy = np.roll(pxy, -1, axis=0) # get next frame value
        vel = ((next_pxy - pxy)/fs)[:-1, :] # the last velocity is not meaningful
        # when we combine this back to the original features, we shift one done,
        # i.e. [p1, p2, ..., pT] combine [_, p2-p1, ...., pT-pT_1]
        # the reason why we shift is that we don't want to leak next position info
        
        acc = (vel[1:,:] - vel[:-1,:])/fs
        out = np.column_stack([moments[2:, :], vel[1:, :], acc])

        return out
    df['enriched'] = df.moments.apply(lambda ms: create_dynamic_features_(ms, fs, n_pl))
    return df['enriched'].values 

