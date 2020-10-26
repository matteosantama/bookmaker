import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd


# Define offensive and defensive features
offense_features = ['PTS', 'AST', 'OREB', 'FGM', 
                    'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA']
defense_features = ['STL', 'DREB', 'BLK', 'TO']
raptor_features = ['raptor_total', 'raptor_box_offense',
                   'raptor_box_defense', 'raptor_onoff_defense',
                   'raptor_onoff_total', 'raptor_offense',
                   'raptor_defense', 'war_total', 'war_reg_season',
                   'pace_impact']

def process_and_write(
        year: int, domain: str, cutoff: int, verbose: str) -> None:
    """Write a single data file labeled '{year}-data.csv to the 
    folder specified by 'domain'.

    Returns:
        None
    """
    if verbose:
        print(f'Analyzing {year} season')

    # Read our raw schedule file and set appropriate index
    sched_path = os.path.join('..', 'data', 'raw', f'{year}-nba-schedule.csv')  
    schedule = pd.read_csv(sched_path, index_col=0, parse_dates=['GAME_DATE'])
    schedule = schedule.set_index(['GAME_ID', 'TEAM_ID'],                       
            verify_integrity=True).sort_values(by=['GAME_DATE', 'GAME_ID'])
    # Now we want to determine if the team is home or away
    schedule['HOME'] = schedule['MATCHUP'].str.split(' ').str[1] == 'vs.'
    assert schedule['HOME'].any()
    #Drop unused columns
    schedule = schedule[['TEAM_NAME', 'GAME_DATE', 'HOME', 'WL', 'PTS']]
    
    # Check two teams for every game
    assert (schedule.groupby(level='GAME_ID').size() == 2).all()
    if verbose:
        print('\tMatchups verified')

    # Now we load the boxscores                                                 
    bs_path = os.path.join('..', 'data', 'raw', f'{year}-nba-boxscores.csv')    
    boxscores = pd.read_csv(bs_path, index_col=0)                                   

    # and set index                                                             
    boxscores = boxscores.set_index(['GAME_ID', 'TEAM_ID', 'PLAYER_ID'],                
            verify_integrity=True).sort_index()      

    # Convert minutes played string to seconds                                  
    def minute_string_to_seconds(x: str) -> int:                                
        if pd.isnull(x):                                                        
            return pd.NA                                                            
        minutes, seconds = map(int, x.split(':'))                               
        return 60 * minutes + seconds  

    # Convert minute string to seconds
    boxscores['SECONDS'] = boxscores['MIN'].map(minute_string_to_seconds)
    # Drop anyone who didn't play
    boxscores = boxscores[~boxscores['SECONDS'].isnull()]
    # Now convert time played to int so we can do operations later
    boxscores['SECONDS'] = boxscores['SECONDS'].astype('int32')

    # Live statistics indexed by (date, playerid) so we can easily look 
    # up a player's current stats for a given day         
    livestats = __compute_live_statistics(schedule, boxscores)
    if verbose:
        print('\tLive season statistics computed')
        
    # Load raptor and join to livestats                                               
    raptor_path = os.path.join('..', 'data', 'raw', 'raptor.csv')    
    raptor = pd.read_csv(raptor_path, index_col=0)  
    livestats = __prepare_raptor(raptor, boxscores, livestats)
    if verbose:
        print('\tRaptor data joined to stats')
    
    game_counts = schedule.groupby('TEAM_ID').cumcount()

    data = []
    for i, (gameid, gamedf) in enumerate(schedule.groupby(level='GAME_ID'), 1):
        assert len(gamedf) == 2                                                 
        team_X, team_Y = gamedf.index.get_level_values(level='TEAM_ID')
        
        games_played_X = game_counts.loc[(gameid, team_X)]
        games_played_Y = game_counts.loc[(gameid, team_Y)]
        # Only proceed if we're deep enough in the season
        if (games_played_X > cutoff and games_played_Y > cutoff):

            players_dict_X = __extract_lineup(
                    boxscores.loc[(gameid, team_X)])       
            players_dict_Y = __extract_lineup(
                    boxscores.loc[(gameid, team_Y)] )   
            
            date_X = schedule.loc[(gameid, team_X)]['GAME_DATE']                      
            date_Y = schedule.loc[(gameid, team_Y)]['GAME_DATE']
            assert date_X == date_Y
            
            stats_series_X = __lookup_stats(players_dict_X, date_X, livestats)
            stats_series_Y = __lookup_stats(players_dict_Y, date_Y, livestats)
    
            # THIS IS WHERE WE ADD FEATURES
            # Make sure that explanatory features are keyed
            # with either 'this' or 'other'
            this_X = pd.concat(
                    [stats_series_X[offense_features]], keys=['this'])
            other_X = pd.concat(
                    [stats_series_Y[defense_features]], keys=['other'])
            this_raptor_X = pd.concat(
                    [stats_series_X[raptor_features]], keys=['this_raptor'])
            other_raptor_X = pd.concat(
                    [stats_series_Y[raptor_features]], keys=['other_raptor'])
            
            row_X = pd.concat([this_X, other_X, this_raptor_X, other_raptor_X])
            row_X[('GAME_ID', '', '')] = gameid
            row_X[('TEAM_ID', '', '')] = team_X
            row_X[('TEAM_PTS', '', '')] = gamedf.loc[(gameid, team_X)]['PTS']
            row_X[('HOME', '', '')] = gamedf.loc[(gameid, team_X)]['HOME']

            this_Y = pd.concat(
                    [stats_series_Y[offense_features]], keys=['this'])
            other_Y = pd.concat(
                    [stats_series_X[defense_features]], keys=['other'])
            this_raptor_Y = pd.concat(
                    [stats_series_Y[raptor_features]], keys=['this_raptor'])
            other_raptor_Y = pd.concat(
                    [stats_series_X[raptor_features]], keys=['other_raptor'])
            
            row_Y = pd.concat([this_Y, other_Y, this_raptor_Y, other_raptor_Y])
            row_Y[('GAME_ID', '', '')] = gameid
            row_Y[('TEAM_ID', '', '')] = team_Y
            row_Y[('TEAM_PTS', '', '')] = gamedf.loc[(gameid, team_Y)]['PTS']
            row_Y[('HOME', '', '')] = gamedf.loc[(gameid, team_Y)]['HOME']

            data.append(row_X.to_dict())
            data.append(row_Y.to_dict())

        if verbose and i % 50 == 0:
            print(f'\t{i} games analyzed')
  
    # The columns here are tuples, but its actually easier to load in this way
    df = pd.DataFrame(data)
    df = df.astype({
        ('GAME_ID', '', ''): 'int32',
        ('TEAM_ID', '', ''): 'int32',
        ('TEAM_PTS', '', ''): 'int32'
    })
    # And set index
    df = df.set_index([('GAME_ID', '', ''), ('TEAM_ID', '', '')])

    writepath = os.path.join('..', 'data', domain, f'{year}-data.csv')
    original = len(df)
    df = df.dropna()

    if verbose:
        print(f'\tWriting {len(df)} scores to {writepath}')

    df.to_csv(writepath)

def __lookup_stats(                                                             
    players: Dict[int, str],                                                    
    date: pd.Timestamp,                                                         
    stats: pd.DataFrame                                                         
) -> pd.DataFrame:                                                              
    """Identify F1, F2, G1, G2, S1, C1 and their stats for a set of players"""  
    # Isolate by date and playerids                                             
    playerstats = stats.loc[date].loc[players]                              
   # Add player position                                                       
    playerstats['POSITION'] = playerstats.index.map(players)
    # Rank by seaonsal seconds played per position                              
    rankings = playerstats.groupby('POSITION')['SECONDS'].rank(                 
            ascending=False, method='first').astype(int).astype(str)
    # Append ranking to position column
    playerstats['POSITION'] = playerstats['POSITION'].str.cat(rankings)         
    # Now we can drop the seconds
    playerstats = playerstats.drop(columns=['SECONDS'])
    # Create a series with a MultiIndex
    playerstats = playerstats.pivot(columns='POSITION').max()
    return playerstats

def __extract_lineup(boxscore: pd.DataFrame) -> Dict[int, str]:        
    """Retrieve starting lineup (plus sub) for a team.                          
                                                                                
    Returns:                                                                    
        Dictionary mapping player ids to position. Two guards, two forwards,    
        one center, and a top sub ('S') who played the most minutes.            
    """                                                                         
    bench_mask = boxscore['START_POSITION'].isnull()                            
    starters = boxscore[~bench_mask]                                           
    bench = boxscore[bench_mask]                                                      
    top_sub = bench['SECONDS'].idxmax()                                         
    lineup = starters['START_POSITION'].to_dict()
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
    stats = stats[offense_features + defense_features + ['SECONDS']]
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

def __prepare_raptor(
        raptor: pd.DataFrame,
        boxscore: pd.DataFrame,
        stats: pd.DataFrame
        ) -> pd.DataFrame:
    """ 
    Take the live statistics dataframe and join it to Raptor Data
    - Boxscore used to retrieve player id's only
    """
    # extract year and find the season
    stats = stats.reset_index()
    stats['season'] = max(pd.DatetimeIndex(stats['GAME_DATE']).year)
    season = stats['season'][0]
    
    # map player names to ids
    boxnamesreset = boxscore.reset_index()
    boxnamesidnames = boxnamesreset[['PLAYER_ID', 'PLAYER_NAME']]
    boxnamesidnames.drop_duplicates(inplace = True)
    
    # left join ids to raptor dataset
    # null values appear in non-analyzed seasons
    raptor_ided = pd.merge(raptor,
                           boxnamesidnames,
                           how = 'left',
                           left_on = "player_name",
                           right_on = "PLAYER_NAME")
    
    # now drop player_id, PLAYER_NAME, and keep only most recent data
    # set the season to the most recent season for merging
    raptor_nocurr = raptor_ided[raptor_ided['season'] != season]
    raptor_nocurr.drop_duplicates('player_id', keep = 'last', inplace = True)
    raptor_nocurr['season'] = season
    raptor_nocurr['season'].astype(str)
    raptor_clean = raptor_nocurr.drop(['player_id', 'PLAYER_NAME'], axis = 1)
    

    combinedstats = pd.merge(stats,
                           raptor_clean,
                           how = 'left',
                           left_on = ['PLAYER_ID', 'season'],
                           right_on = ['PLAYER_ID', 'season'])
    combinedstats.set_index(['GAME_DATE', 'PLAYER_ID'], inplace = True)
    combinedstats.drop('season', inplace = True, axis = 1)
    
    # fill rookie NA values
    combinedstats['poss'].fillna(np.mean(combinedstats['poss']), inplace = True)
    combinedstats['mp'].fillna(np.mean(combinedstats['mp']), inplace = True)
    combinedstats.fillna(0, inplace = True)
    
    # To check out all the players that don't have raptor stats 
    # This should be all rookies
#     nulls = combinedstats[combinedstats['poss'].isnull()]
#     nulls.drop_duplicates('PLAYER_ID', inplace = True)
    
#     nulls = nulls[['PLAYER_ID', 'season', 'raptor_total']]
#     check_nulls = pd.merge(raptor_ided, nulls,
#                            how = 'inner',
#                            left_on = 'PLAYER_ID',
#                            right_on = 'PLAYER_ID').reset_index()
    
#     check_nulls = check_nulls[['PLAYER_ID',
#                                'PLAYER_NAME',
#                                'season_x',
#                                'season_y',
#                                'raptor_total_y']]
    return combinedstats

if __name__=='__main__':
    parser = argparse.ArgumentParser(description=(
            'Combine nba-schedule file with nba-boxscores file to construct '
            'a new .csv file that aligns game data with player data'))

    parser.add_argument('domain', choices=['train', 'dev', 'test'],
            help='Where to find the nba-*.csv files')
    
    parser.add_argument('--years', type=int, choices=range(2015, 2021),
            dest='years', nargs='*', help=('A season is identified by its '
                'closing year. You can specify multiple years'))

    parser.add_argument('-c', type=int, default=3, dest='cutoff', 
            choices=range(1, 25), help=
            'Number of games a team must have played in a season')

    parser.add_argument('-v', dest='verbose', action='store_true', 
            help='Display verbose output flag')

    args = parser.parse_args()

    for y in args.years:
        process_and_write(y, args.domain, args.cutoff, args.verbose)
