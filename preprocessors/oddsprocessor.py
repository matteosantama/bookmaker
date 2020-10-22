import argparse
import os
import re

import pandas as pd


# Mapping to canonical team names
teams = {
    'Atlanta':          'Atlanta Hawks',
    'Boston':           'Boston Celtics',
    'Brooklyn':         'Brooklyn Nets',
    'Charlotte':        'Charlotte Hornets',
    'Chicago':          'Chicago Bulls',
    'Cleveland':        'Cleveland Cavaliers',
    'Dallas':           'Dallas Mavericks',
    'Denver':           'Denver Nuggets',
    'Detroit':          'Detroit Pistons',
    'GoldenState':      'Golden State Warriors',
    'Houston':          'Houston Rockets',
    'Indiana':          'Indiana Pacers',
    'LAClippers':       'Los Angeles Clippers',
    'LA Clippers':      'Los Angeles Clippers',
    'LALakers':         'Los Angeles Lakers',
    'Memphis':          'Memphis Grizzlies',
    'Miami':            'Miami Heat',
    'Milwaukee':        'Milwaukee Bucks',
    'Minnesota':        'Minnesota Timberwolves',
    'NewOrleans':       'New Orleans Pelicans',
    'NewYork':          'New York Knicks',
    'OklahomaCity':     'Oklahoma City Thunder',
    'Oklahoma City':    'Oklahoma City Thunder',
    'Orlando':          'Orlando Magic',
    'Philadelphia':     'Philadelphia 76ers',
    'Phoenix':          'Phoenix Suns',
    'Portland':         'Portland Trail Blazers',
    'Sacramento':       'Sacramento Kings',
    'SanAntonio':       'San Antonio Spurs',
    'Toronto':          'Toronto Raptors',
    'Utah':             'Utah Jazz',
    'Washington':       'Washington Wizards',
}


def process_and_write(year: int, domain: str, verbose: bool) -> None:
    """Load odds Excel File from /data/raw/, and stores a processed
    dataframe in either /data/train/, /data/dev/, or /data/test/
    """
    inpath = os.path.join(
        '..', 'data', 'raw', f'odds-{year - 1}-{year - 2000}.xlsx')
    raw = pd.read_excel(inpath)
    # First we have to fix the date because no year info is included
    if verbose:
        print('\tParsing monthyear dates')
        
    new_year = False
    for i, monthday in raw['Date'].iteritems():
        stringified = str(monthday)
        day = int(stringified[-2:])
        month = int(stringified[:-2])
        if month == 1:
            new_year = True
        ts_year = year if new_year else year - 1
        ts = pd.Timestamp(year=ts_year, month=month, day=day)
        raw.at[i, 'Date'] = ts

    # Next we standarize the team names
    if verbose:
        print('\tUpdating team names')
        
    raw['Team'] = raw['Team'].map(teams)
   
    # If we read these columns in as strings, we need to replace and convert
    regex = re.compile('pk', flags=re.I)
    raw[['Open', 'Close']] = raw[['Open', 'Close']].replace(
            regex, 0., regex=True)

    df = pd.DataFrame(index=pd.RangeIndex(len(raw)), columns=[
        'GAME_DATE', 'TEAM_NAME', 'pts', 'ou_open',
        'ou_close', 'ml', 'spread_open', 'spread_close'
    ])
    for i in range(0, len(df), 2):
        # Select two teams at a time
        row_a = raw.iloc[i]
        row_b = raw.iloc[i + 1]
        assert row_a.at['Date'] == row_b.at['Date']

        home = row_a if row_a.at['VH'] == 'H' else row_b
        away = row_b if row_b.at['VH'] == 'V' else row_a
        assert not home.equals(away)

        h_open, h_close = home.at['Open'], home.at['Close']
        a_open, a_close = away.at['Open'], away.at['Close']

        # In terms of structure, the odds files are a mess.
        # We have to do some smart guesswork to pull all the odds
        ou_open = max(h_open, a_open)
        ou_close = max(h_close, a_close)

        spread_open = min(h_open, a_open)
        spread_close = min(h_close, a_close)

        home_is_fav = home['ML'] <= away['ML']

        for offset, team in enumerate([home, away]):
            df.at[i + offset, 'GAME_DATE'] = team['Date']
            df.at[i + offset, 'TEAM_NAME'] = team['Team']
            df.at[i + offset, 'pts'] = team['Final']
            df.at[i + offset, 'ml'] = team['ML']
            df.at[i + offset, 'ou_open'] = ou_open
            df.at[i + offset, 'ou_close'] = ou_close
            df.at[i + offset, 'spread_open'] = spread_open
            df.at[i + offset, 'spread_close'] = spread_close 
            # Spreads need to take into account who the favorite is
            if team.equals(home) and home_is_fav:
                df.at[i + offset, 'spread_open'] *= -1
                df.at[i + offset, 'spread_close'] *= -1

    # At this point, the odds data has been cleaned. Now we need to 
    # add NBA game and team ids
    schedulefilename = f'{year}-nba-schedule.csv'
    schedulepath = os.path.join('..', 'data', 'raw', schedulefilename)
    schedule = pd.read_csv(
            schedulepath, index_col=0, parse_dates=['GAME_DATE'])
    # In 2016, the NBA calls them the 'LA Clippers', so we want to
    # map that to the full name as well
    schedule['TEAM_NAME'] = schedule['TEAM_NAME'].apply(
            lambda x: x if x in teams.values() else teams[x])
    
    if verbose:
        print('\tJoining NBA gameid and teamid')
    join_keys = ['TEAM_NAME', 'GAME_DATE']
    schedule = schedule.set_index(join_keys)
    schedule = schedule[['GAME_ID', 'TEAM_ID']]

    df = df.set_index(join_keys)
    # We only keep odds that are in the schedule file
    df = df.join(schedule, how='right')
    # Make sure we didn't lose a team along the way
    for k, v in teams.items():
        assert v in df.index.get_level_values(0), f'MISSING {v} in {year}'

    outpath = os.path.join('..', 'data', domain, f'{year}-odds.csv')
    df.to_csv(outpath)
    if verbose:
        print(f'\t{outpath} written')


if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='Process Excel files from Sports Book Review Online.'
    )

    parser.add_argument('domain', choices=['train', 'dev', 'test'],
        help='Whether to store in data/train/, data/dev/,  or data/test/')
    
    parser.add_argument('--years', type=int, choices=range(2015, 2021), 
            dest='years', nargs='*', help=('Seasons are identified by their '
            'ending year. You can specify multiple 4 digit years'))

    parser.add_argument('-v', dest='verbose',
        help='Flag to display verbose output', action='store_true')
    
    args = parser.parse_args()
    
    for year in args.years:
        process_and_write(year, args.domain, args.verbose)
