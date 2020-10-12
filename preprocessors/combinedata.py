import argparse
import os
import itertools as it
from typing import Dict, List
import pdb

import pandas as pd


def process_and_write(year: int, domain: str, verbose: str) -> None:
    """Write a single file to /`domain`"""
    sched_path = os.path.join('..', 'data', 'raw', f'{year}-nba-schedule.csv')  
    sched_df = pd.read_csv(sched_path, index_col=0, parse_dates=['GAME_DATE'])

    # Cut down on number of columns to preserve memory                          
    sched_df = sched_df.set_index(['GAME_ID', 'TEAM_ID'],                       
            verify_integrity=True).sort_values(by=['GAME_DATE', 'GAME_ID'])                                       
    sched_df = sched_df[['TEAM_NAME', 'GAME_DATE', 'MATCHUP', 'WL', 'PTS']]
    
    # Check two teams for every game
    assert (sched_df.groupby(level='GAME_ID').size() != 2).sum() == 0
    if verbose:
        print('\tMatchups verified')

    # Now we load the boxscores                                                 
    bs_path = os.path.join('..', 'data', 'raw', f'{year}-nba-boxscores.csv')    
    bs_df = pd.read_csv(bs_path, index_col=0)                                   

    # and set index                                                             
    bs_df = bs_df.set_index(['GAME_ID', 'TEAM_ID', 'PLAYER_ID'],                
            verify_integrity=True).sort_index()      

    # Convert minutes played string to seconds                                  
    def minute_string_to_seconds(x: str) -> int:                                
        if pd.isnull(x):                                                        
            return pd.NA                                                            
        minutes, seconds = map(int, x.split(':'))                               
        return 60 * minutes + seconds  

    # Convert minute string to seconds
    bs_df['SECONDS'] = bs_df['MIN'].map(minute_string_to_seconds)
    # Drop anyone who didn't play
    bs_df = bs_df[~bs_df['SECONDS'].isnull()]
    # Now make an int so we can do operations later
    bs_df['SECONDS'] = bs_df['SECONDS'].astype('int32')

    # So we can easily look up a player's current stats for a given day         
    livestats = __compute_live_statistics(sched_df, bs_df)                      
    if verbose:
        print('\tLive season statistics computed')

    # Here we define our parameters of interest                                 
    features = [                                                                
            'PTS', 'TO', 'BLK', 'STL', 'AST', 'REB',                            
            'FTM', 'FTA', 'FGM', 'FGA', 'FG3M', 'FG3A'                          
    ]                                                                           
    roles = ['G1', 'G2', 'F1', 'F2', 'C1', 'S1']                                  
    product = it.product(                                                       
            ['this', 'other'],                                                  
            roles,                                                              
            features                                                            
    )                                                                           
    product = map(lambda x: '.'.join(x), product)
    cols = ['TEAM_PTS'] + list(product)
    df = pd.DataFrame(index=sched_df.index, columns=cols)            

    # We drop the first game for every team, since we don't
    # have player stats yet
    team_game_count = sched_df.groupby(level='TEAM_ID').cumcount()              
    mask_by_team = team_game_count >= 1                                         
    mask_by_game = mask_by_team.groupby(level='GAME_ID').all()                  
    df = df.loc[mask_by_team & mask_by_game]
    if verbose:
        drop_count = (~(mask_by_team & mask_by_game)).sum()
        print(f'\t{drop_count} games are the first of the season')
        print('\tConstructing final DataFrame...')

    for i, (gameid, gamedf) in enumerate(df.groupby(level='GAME_ID'), 1):
        assert len(gamedf) == 2                                                 
        team_a, team_b = gamedf.index.get_level_values(level='TEAM_ID')
        try:
            a_players = __lineup(bs_df.loc[(gameid, team_a)])       
            b_players = __lineup(bs_df.loc[(gameid, team_b)])   

            a_date = sched_df.loc[(gameid, team_a)]['GAME_DATE']                      
            b_date = sched_df.loc[(gameid, team_b)]['GAME_DATE']
            assert a_date == b_date
    
            a_stats = __lookup_stats(a_players, a_date, features, livestats)                  
            b_stats = __lookup_stats(b_players, b_date, features, livestats)
    
            flat_a = __flatten(a_stats)                                             
            flat_b = __flatten(b_stats)                                             

            a_row = __combine(flat_a, flat_b)    
            b_row = __combine(flat_b, flat_a)
    
            a_row['TEAM_PTS'] = sched_df.loc[(gameid, team_a)]['PTS']
            b_row['TEAM_PTS'] = sched_df.loc[(gameid, team_b)]['PTS']

            df.loc[(gameid, team_a)] = a_row                                        
            df.loc[(gameid, team_b)] = b_row
        
        except RuntimeError as e:
            # Some games have corrupted data
            pass

        if verbose and i % 50 == 0:
            print(f'\t{i} games analyzed')
    
    writepath = os.path.join('..', 'data', domain, f'{year}-data.csv')
    original = len(df)
    df = df.dropna()

    if verbose:
        print(f'\t{len(df)} out of {original} games preserved')
        print(f'\tWriting {writepath}')

    df.to_csv(writepath)


def __combine(
        this: Dict[str, float], other: Dict[str, float]) -> Dict[str, float]:
    """Inputs to this should be the outputs of `__flatten()`"""
    d = {}
    for k, v in this.items():
        d[f'this.{k}'] = v
    for k, v in other.items():
        d[f'other.{k}'] = v
    return d

def __flatten(stats: pd.DataFrame) -> Dict[str, float]:                         
    """Flatten a 2d DataFrame into a dictionary that represents one row"""      
    d = stats.to_dict(orient='index')                                         
    new_dict = {}                                                               
    for pos_key, stats_dict in d.items():                                       
        for stat_key in stats_dict:                                             
            new_dict[f'{pos_key}.{stat_key}'] = stats_dict[stat_key]            
    return new_dict

def __lookup_stats(                                                             
    players: Dict[int, str],                                                    
    date: pd.Timestamp,                                                         
    features: List[str],                                                        
    stats: pd.DataFrame                                                         
) -> pd.DataFrame:                                                              
    """Identify F1, F2, G1, G2, S1, C1 and their stats for a set of players"""  
    # Isolate by date and playerids                                             
    playerstats = stats.loc[date].loc[players]                              
    # Add player position                                                       
    playerstats['POSITION'] = playerstats.index.map(players)
    # Rank by seaonsal seconds played per position                              
    try:
        rankings = playerstats.groupby('POSITION')['SECONDS'].rank(                 
            ascending=False).astype(int).astype(str)
    except ValueError:
        pdb.set_trace()
    # Append ranking to position column                                         
    playerstats['POSITION'] = playerstats['POSITION'].str.cat(rankings)         
    # Index by unique position + ranking
    filtered = playerstats.set_index('POSITION')[features]
    
    truth = pd.Series(
        {'F1': 1, 'F2': 1, 'G1': 1, 'G2': 1, 'C1': 1, 'S1': 1}, 
        name='POSITION'
    ).sort_index()
    counts = filtered.index.value_counts().sort_index()
    try:
        equals = (truth == counts).all()
    except ValueError:
        equals = False
    hasnull = filtered.isnull().values.any()
    
    if not equals:
        msg = f'Missing starters: {filtered}'
        raise RuntimeError(msg)
    if hasnull:
        msg = f'Encountered NAs in {filtered}'
        raise RuntimeError(msg)
    return filtered

def __lineup(boxscore: pd.DataFrame) -> Dict[int, str]:        
    """Retrieve starting lineup (plus sub) for a team.                          
                                                                                
    Returns:                                                                    
        Dictionary mapping player ids to position. Two guards, two forwards,    
        one center, and a top sub ('S') who played the most minutes.            
    """                                                                         
    bench_mask = boxscore['START_POSITION'].isnull()                            
    starters = boxscore[~bench_mask]                                           
    bench = boxscore[bench_mask]                                                      
                                                                                
    top_sub = bench['SECONDS'].idxmax()                                         
    lineup = dict(starters['START_POSITION'])                                   
    lineup[top_sub] = 'S'                                                        
    return lineup

def __compute_live_statistics(                                                  
        schedule: pd.DataFrame, boxscore: pd.DataFrame) -> pd.DataFrame:        
    """This function returns a dataframe with a (date, id) index for every      
    date and every player in the season. The dates are shifted back a day,      
    so we can get stats for a game without including that game's info. Every    
    column is also standardized to per-48 minutes, except for the seconds       
    column which is cumulative.                                                 
    """                                                                  
    # First we copy the boxscores                
    stats = boxscore.copy()
    
    # Now we can drop unnecessary columns
    stats = stats.drop(columns=[                                      
            'TEAM_ABBREVIATION', 'PLAYER_NAME', 'TEAM_CITY', 'COMMENT',         
            'FG_PCT', 'FG3_PCT', 'FT_PCT', 'MIN', 'START_POSITION',             
            'PLUS_MINUS'                                                        
    ])    
    # Then we add dates by joining with the schedule df                         
    stats = stats.join(schedule['GAME_DATE'], on=['GAME_ID', 'TEAM_ID'])        
    # Then we reset the index and reindex on date and player_id                 
    stats = stats.reset_index().set_index(['GAME_DATE', 'PLAYER_ID'],           
            verify_integrity=True).drop(columns=['GAME_ID', 'TEAM_ID'])  
    # We have to make sure the dates are in order though                        
    stats = stats.sort_index(level='GAME_DATE')
    
    # Not done yet though, we compute live stats for every player on            
    # every day. Then we can easily look up stats per player per day            
    players = stats.index.get_level_values('PLAYER_ID').unique()                
    season_open = stats.index.get_level_values('GAME_DATE').min()               
    season_close = stats.index.get_level_values('GAME_DATE').max()              
                                                                                
    mi = pd.MultiIndex.from_product([                                           
            pd.date_range(season_open, season_close, name='GAME_DATE'),         
            players                                                             
    ])      
    # reindex and sum by player
    stats = stats.reindex(mi).fillna(0).groupby(level='PLAYER_ID').cumsum()
                                                                                
    # and shift back a day so a day's stats actaully represent                  
    # stats current as of the night before                                      
    stats = stats.groupby(level='PLAYER_ID').shift(1)
                                                                                
    # Finally, we can normalize per 48
    seconds = stats['SECONDS']
    stats = stats.div(seconds / (60 * 48), axis=0)                              
    stats['SECONDS'] = seconds
    # Dividing by zero might have introduced NAs
    stats = stats.fillna(0)
    return stats


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=(
            'Combine nba-schedule file with nba-boxscores file to construct '
            'a new .csv file that aligns game data with player data'))

    parser.add_argument('year', type=int, choices=range(2015, 2021),
            help='A season is identified by its closing year')

    parser.add_argument('domain', choices=['train', 'dev', 'test'],
            help='Where to find the nba-*.csv files')

    parser.add_argument('-v', dest='verbose', action='store_true', 
            help='Display verbose output flag')

    args = parser.parse_args()

    process_and_write(args.year, args.domain, args.verbose)
