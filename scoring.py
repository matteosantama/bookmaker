from typing import Tuple, List, Union

import pandas as pd
import numpy as np

import torch.nn as nn

import attn_dl

def score(
        model: nn.Sequential, 
        bet: str, 
        category: str,
        buffer: float=1.
) -> Tuple[int, int, int]:
    """
    This function handles some preliminary data loading tasks before redirecting to 
    the more specific _score_spread() and _score_overunder() functions.
    
    Parameters:
        model: The ANN object with learned parameters
        bet: Either 'spread' or 'over/under'
        category: Either 'open' or 'close'
        buffer: How much our model must contradict the odds to place a bet.
                Otherwise we just 'push' and don't take any action.
        
    Returns:
        A three-tuple consisting of (bets won, bets lost, bets pushed)
    """
    if buffer < 0:
        raise RuntimeError(f' {buffer} makes no sense')
        
    if category not in {'open', 'close'}:
        raise RuntimeError(f" {category} not valid, must be in ['open', 'close']")
    
    i_test, x_test, y_test = attn_dl.load_vectorized_data('test')
    y_predicted = model(x_test)    

    # pd.Series are way easier to handle than tensors
    y_hat = pd.Series(y_predicted.detach().squeeze(), index=i_test)
    y = pd.Series(y_test.detach().squeeze(), index=i_test)

    odds = attn_dl.load_odds_data('test').dropna()
    intersection = odds.index.intersection(i_test)
    # Drop some games, since our prepared NBA data doesn't contain
    # every game in the season.
    odds = odds.loc[intersection]
    y_hat = y_hat.loc[intersection]
    y = y.loc[intersection]
    
    # Check that both data sources agree on the points scored
    assert (y == odds['pts']).all()

    if bet == 'over/under':
        return _score_overunder(model, odds[f'ou_{category}'], 
                                y, y_hat, buffer)
    
    elif bet == 'spread':
        return _score_spread(model, odds[f'spread_{category}'], 
                             y, y_hat, buffer)
        
    else:
        msg = f"{bet} not valid, must be in ['over/under', 'spread']"
        raise RuntimeError(msg)

def _score_spread(
        model: nn.Sequential, 
        odds: pd.Series, 
        y: pd.Series, 
        y_hat: pd.Series,
        buffer: float=1.
) -> Tuple[int, int, int]:
    """
    Given a 'buffer', determine how many 'spread' bets we would have
    placed and won.
    
    Parameters:
        model: ANN with trained model parameters
        odds: A slice of the original df with either 'open' or 'close'
        y: Actual score outcomes per team
        y_hat: Predicted score outcomes per team
        buffer: A measure of how much our model must contradict the odds
                in order to place a bet.
                
    Returns:
        A three-tuple consisting of (games won, games lost, no action taken)
    """
    # We need to divided be -2 here because the favorite is always given a negative
    # spread, but we actually want a positive difference with the opponent
    bookie = odds.groupby(level='GAME_ID').diff().dropna() / -2
    actual_spreads = y.groupby(level='GAME_ID').diff().dropna()
    model_spreads = y_hat.groupby(level='GAME_ID').diff().dropna()
    
    long = model_spreads - buffer > bookie
    won_long = long & (actual_spreads > bookie)
    
    short = model_spreads + buffer < bookie
    won_short = short & (actual_spreads < bookie)
    
    push = ~long & ~short
    
    won = won_long.sum() + won_short.sum()
    lost = long.sum() + short.sum() - won
    pushed = push.sum()
    
    return won, lost, pushed

def _score_overunder(
        model: nn.Sequential, 
        odds: pd.Series, 
        y: pd.Series, 
        y_hat: pd.Series,
        buffer: float=1.
) -> Tuple[int, int, int]:
    """
    Given a 'buffer', determine how many over/under bets we would have
    placed and won.
    
    Parameters:
        model: ANN with trained model parameters
        odds: A slice of the original df with either 'open' or 'close'
        y: Actual score outcomes per team
        y_hat: Predicted score outcomes per team
        buffer: A measure of how much our model must contradict the odds
                in order to place a bet.
                
    Returns:
        A three-tuple consisting of (games won, games lost, no action taken)
    """
    bookie = odds.groupby(level='GAME_ID').first()
    actual_totals = y.groupby(level='GAME_ID').sum()
    model_totals = y_hat.groupby(level='GAME_ID').sum()
    long = model_totals + buffer < bookie
    won_long = long & (actual_totals < bookie)
    
    short = model_totals - buffer > bookie
    won_short = short & (actual_totals > bookie)
    
    push = ~long & ~short
    
    won = won_long.sum() + won_short.sum()
    lost = long.sum() + short.sum() - won
    pushed = push.sum()
    
    return won, lost, pushed