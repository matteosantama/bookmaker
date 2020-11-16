import os
import glob
from typing import Tuple, List, Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch

os.environ['KMP_DUPLICATE_LIB_OK']='True'


def load_vectorized_data(_type: str):
    """
    Loads the data, joining players together as a vector of player-specific statistics
    (Instead of single vector for all statistics in every row)
    """
    df = pd.DataFrame()
    data_path = os.path.join('data', _type, '*-data-raptor.csv')
    for fp in glob.glob(data_path):
        season = pd.read_csv(fp)
        # Convert string tuples to actual tuples
        season = season.rename(columns=lambda x: eval(x))
        season = season.set_index([('GAME_ID', '', ''), ('TEAM_ID', '', '')])
        df = df.append(season)
    
    #list of columns by position
    ptypes = ['C1', 'F1', 'F2', 'G1', 'G2', 'S1']
    positions = [[column for column in df.columns if playertype in column] for playertype in ptypes]
    new = df
    # Join players together as a vector of statistics
    for i in range(len(ptypes)):
        new[ptypes[i]] = new[positions[i]].values.tolist()
        new[ptypes[i]] = new[ptypes[i]].apply(lambda x: np.array(x))
    featurelen = len(new[ptypes[0]].iloc[0])
    new[('HOME', '', '')] = new[('HOME', '', '')].apply(lambda x: np.array([x]*featurelen))
    playervec_df = new[['C1', 'F1', 'F2', 'G1', 'G2', 'S1', ('TEAM_PTS', '', ''), ('HOME', '', '')]]

    outcome_col = ('TEAM_PTS', '', '')
    features = playervec_df[playervec_df.columns.difference([outcome_col])]

    # Need to keep this a DataFrame so our dimensions work in PyTorch
    scores = playervec_df[[outcome_col]]

    n_features = len(features.columns)
    n_output = len(scores.columns)
    msg = 'Uh oh, you might be losing features!'
    assert n_features + n_output == len(playervec_df.columns), msg
    
    # stack the arrays together on axis = 1 so torch can convert it
    test = np.array([np.stack(example, axis = 1) for example in features.values])

    features = torch.from_numpy(test)
    scores = torch.from_numpy(scores.to_numpy())
    simple_index = df.index.rename(['GAME_ID', 'TEAM_ID'])
    
    # flip so each row is a f
    features = features.transpose(1,2)
    
    return simple_index, features, scores


def load_odds_data(_type: str) -> pd.DataFrame:
    """
    Load the odds DataFrame for a certain domain.
    
    Parameters:
        _type: Choose from 'train', 'dev', or 'test'
        
    Returns:
        A pd.DataFrame indexed by (GAME_ID, TEAM_ID)
    """
    if _type not in {'train', 'dev', 'test'}:
        msg = f"{_type} not supported. Try 'train', 'dev', or 'test'."
        raise RuntimeError(msg)
        
    df = pd.DataFrame()
    data_path = os.path.join('data', _type, '*-odds.csv')
    for fp in glob.glob(data_path):
        season_df = pd.read_csv(fp, index_col=[0, 1])
        df = df.append(season_df)
        
    df = df.reset_index()
    df = df.set_index(['GAME_ID', 'TEAM_ID'])
    return df